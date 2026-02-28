#!/usr/bin/env python3
"""
Rebuild DeepFilterNet-style sidecars for a personalization HDF5 file.

Outputs:
  - <hdf5_path>.speaker_ids.json
  - <hdf5_path>.speaker_index.json

This script does NOT modify HDF5 audio content.
"""

import argparse
import json
import os
import sys
from array import array

import h5py as h5


def _key_from_entry(entry: dict, audio_key: str, base_audio_dir: str) -> str:
    """Derive HDF5 dataset key from manifest entry.

    Must match ExportPersonalizationArtifacts._entry_to_key_and_speaker in
    sdp/processors/modify_manifest/data_to_data.py exactly, so that keys
    built from manifest_val.json match the keys written into the HDF5.
    """
    rel_path = entry[audio_key]
    audio_path = os.path.join(base_audio_dir, rel_path)
    key = os.path.relpath(audio_path, base_audio_dir).replace(os.sep, "_")
    return key


def _build_key_to_speaker(
    manifest_path: str,
    base_audio_dir: str,
    split: str = None,
    set_key: str = "set",
    speaker_key: str = "speaker",
    audio_key: str = "audio_filepath",
) -> dict:
    """Build key -> speaker from manifest (last entry wins for duplicate keys)."""
    key_to_speaker = {}
    with open(manifest_path, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            if split is not None and entry.get(set_key) != split:
                continue
            key = _key_from_entry(entry, audio_key=audio_key, base_audio_dir=base_audio_dir)
            key_to_speaker[key] = str(entry[speaker_key])
    return key_to_speaker


def _manifest_entries_in_order(
    manifest_path: str,
    base_audio_dir: str,
    split: str = None,
    set_key: str = "set",
    speaker_key: str = "speaker",
    audio_key: str = "audio_filepath",
):
    """Yield (key, speaker) in manifest order (for order-based matching)."""
    with open(manifest_path, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            if split is not None and entry.get(set_key) != split:
                continue
            key = _key_from_entry(entry, audio_key=audio_key, base_audio_dir=base_audio_dir)
            yield key, str(entry[speaker_key])


def rebuild_sidecars(
    manifest_path: str,
    hdf5_path: str,
    base_audio_dir: str,
    split: str = None,
    set_key: str = "set",
    speaker_key: str = "speaker",
    audio_key: str = "audio_filepath",
    hdf5_group: str = "speech",
):
    """
    Rebuild speaker_ids.json and speaker_index.json from manifest + HDF5.

    Iteration order is the HDF5 group key order so that index_to_speaker[i]
    matches the i-th key in the HDF5 (as expected by DeepFilterNet).
    """
    hdf5_filename = os.path.basename(hdf5_path)
    sidecar_ids_path = hdf5_path + ".speaker_ids.json"
    sidecar_index_path = hdf5_path + ".speaker_index.json"

    sidecar_ids_tmp = sidecar_ids_path + ".tmp"
    sidecar_index_tmp = sidecar_index_path + ".tmp"

    key_to_speaker = _build_key_to_speaker(
        manifest_path, base_audio_dir, split=split, set_key=set_key,
        speaker_key=speaker_key, audio_key=audio_key,
    )

    speaker_to_indices = {}
    index_to_speaker_list = []
    hdf5_keys_ordered = []  # keys in HDF5 iteration order
    keys_missing_speaker = 0

    with h5.File(hdf5_path, "r", libver="latest") as f:
        if hdf5_group not in f:
            raise KeyError(f"HDF5 group '{hdf5_group}' not found in {hdf5_path}")
        grp = f[hdf5_group]

        for key in grp.keys():
            hdf5_keys_ordered.append(key)

    keys_in_hdf5 = len(hdf5_keys_ordered)

    # Order-based matching: HDF5 uses numeric string keys "0","1",... (e.g. from another pipeline).
    if keys_in_hdf5 > 0 and all(k.isdigit() for k in hdf5_keys_ordered):
        manifest_ordered = list(_manifest_entries_in_order(
            manifest_path, base_audio_dir, split=split, set_key=set_key,
            speaker_key=speaker_key, audio_key=audio_key,
        ))
        if len(manifest_ordered) != keys_in_hdf5:
            raise ValueError(
                f"Order-based matching: manifest has {len(manifest_ordered)} entries for this split "
                f"but HDF5 has {keys_in_hdf5} keys. Counts must match."
            )
        index_to_speaker_list = [speaker for _, speaker in manifest_ordered]
        key_to_speaker = {hdf5_keys_ordered[i]: manifest_ordered[i][1] for i in range(keys_in_hdf5)}
        speaker_to_indices = {}
        for idx, (_, speaker) in enumerate(manifest_ordered):
            speaker_to_indices.setdefault(speaker, array("I")).append(idx)
        keys_missing_speaker = 0
    else:
        # Path-based matching: key from manifest (same formula as ExportPersonalizationArtifacts).
        for idx, key in enumerate(hdf5_keys_ordered):
            speaker = key_to_speaker.get(key)
            if speaker is None:
                keys_missing_speaker += 1
            index_to_speaker_list.append(speaker)
            if speaker is not None:
                speaker_to_indices.setdefault(speaker, array("I")).append(idx)

    total_samples = len(index_to_speaker_list)

    # When no keys matched, help the user debug path format mismatch
    if keys_missing_speaker == keys_in_hdf5 and keys_in_hdf5 > 0:
        manifest_sample = list(key_to_speaker.keys())[:3] if key_to_speaker else []
        hdf5_sample = hdf5_keys_ordered[:3]
        print(
            "WARNING: No manifest entries matched any HDF5 key (speaker_count=0).\n"
            "Keys are derived the same way as ExportPersonalizationArtifacts (data_to_data.py).\n"
            "Sample keys from manifest (first 3):",
            manifest_sample or "(none)",
            "\nSample keys from HDF5 (first 3):",
            hdf5_sample,
            "\nPass the same --base-audio-dir that was used when the HDF5 was created\n"
            "(e.g. audio_source_dir / base_audio_dir from the export config).",
            file=sys.stderr,
        )

    try:
        with open(sidecar_ids_tmp, "w", encoding="utf8") as ids_out:
            ids_out.write("{")
            ids_out.write(json.dumps(hdf5_filename))
            ids_out.write(":{")
            first_pair = True
            for i, key in enumerate(hdf5_keys_ordered):
                speaker = index_to_speaker_list[i] if i < len(index_to_speaker_list) else None
                if not first_pair:
                    ids_out.write(",")
                first_pair = False
                ids_out.write(json.dumps(key))
                ids_out.write(":")
                ids_out.write(json.dumps(speaker) if speaker is not None else "null")
            ids_out.write("}}")

        with open(sidecar_index_tmp, "w", encoding="utf8") as fout:
            fout.write("{")
            fout.write("\"speaker_to_indices\":{")
            first_speaker = True
            for speaker, idxs in speaker_to_indices.items():
                if not first_speaker:
                    fout.write(",")
                first_speaker = False
                fout.write(json.dumps(speaker))
                fout.write(":[")
                for i, sample_idx in enumerate(idxs):
                    if i > 0:
                        fout.write(",")
                    fout.write(str(int(sample_idx)))
                fout.write("]")
            fout.write("},")

            fout.write("\"index_to_speaker\":[")
            for i, speaker in enumerate(index_to_speaker_list):
                if i > 0:
                    fout.write(",")
                fout.write(json.dumps(speaker) if speaker is not None else "null")
            fout.write("],")
            fout.write(f"\"total_samples\":{total_samples}")
            fout.write("}")

        # Atomic replace only when both sidecars are fully written.
        os.replace(sidecar_ids_tmp, sidecar_ids_path)
        os.replace(sidecar_index_tmp, sidecar_index_path)
    finally:
        if os.path.exists(sidecar_ids_tmp):
            os.remove(sidecar_ids_tmp)
        if os.path.exists(sidecar_index_tmp):
            os.remove(sidecar_index_tmp)

    print(
        "Rebuilt sidecars:",
        json.dumps(
            {
                "hdf5": hdf5_path,
                "total_samples": total_samples,
                "keys_in_hdf5": keys_in_hdf5,
                "keys_missing_speaker": keys_missing_speaker,
                "speaker_count": len(speaker_to_indices),
            },
            indent=2,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Rebuild speaker sidecars from manifest + HDF5 keys.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL (split manifest or per-split manifest).")
    parser.add_argument("--hdf5", required=True, help="Target HDF5 file path.")
    parser.add_argument(
        "--base-audio-dir",
        required=True,
        help="Base audio directory; must be the same as base_audio_dir / audio_source_dir used when ExportPersonalizationArtifacts created the HDF5 (see data_to_data.py).",
    )
    parser.add_argument("--split", default=None, help="Optional split to filter by (e.g., train/test/val).")
    parser.add_argument("--set-key", default="set", help="Manifest split field name.")
    parser.add_argument("--speaker-key", default="speaker", help="Manifest speaker field name.")
    parser.add_argument("--audio-key", default="audio_filepath", help="Manifest audio path field name.")
    parser.add_argument("--hdf5-group", default="speech", help="HDF5 group containing datasets.")
    args = parser.parse_args()

    rebuild_sidecars(
        manifest_path=args.manifest,
        hdf5_path=args.hdf5,
        base_audio_dir=args.base_audio_dir,
        split=args.split,
        set_key=args.set_key,
        speaker_key=args.speaker_key,
        audio_key=args.audio_key,
        hdf5_group=args.hdf5_group,
    )


if __name__ == "__main__":
    main()

