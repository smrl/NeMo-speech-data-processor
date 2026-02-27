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
from array import array

import h5py as h5


def _key_from_entry(entry: dict, audio_key: str, base_audio_dir: str) -> str:
    rel_path = entry[audio_key]
    audio_path = os.path.join(base_audio_dir, rel_path)
    return os.path.relpath(audio_path, base_audio_dir).replace(os.sep, "_")


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

        # Iterate HDF5 keys in their natural order so indices match HDF5 iteration.
        for idx, key in enumerate(grp.keys()):
            hdf5_keys_ordered.append(key)
            speaker = key_to_speaker.get(key)
            if speaker is None:
                keys_missing_speaker += 1
            index_to_speaker_list.append(speaker)
            if speaker is not None:
                speaker_to_indices.setdefault(speaker, array("I")).append(idx)

    total_samples = len(index_to_speaker_list)
    keys_in_hdf5 = len(hdf5_keys_ordered)

    try:
        with open(sidecar_ids_tmp, "w", encoding="utf8") as ids_out:
            ids_out.write("{")
            ids_out.write(json.dumps(hdf5_filename))
            ids_out.write(":{")
            first_pair = True
            for key in hdf5_keys_ordered:
                speaker = key_to_speaker.get(key)
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
    parser.add_argument("--base-audio-dir", required=True, help="Base audio directory used to derive HDF5 keys.")
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

