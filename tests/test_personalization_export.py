import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import soundfile as sf
import h5py as h5

from sdp.processors.modify_manifest.data_to_data import ExportPersonalizationArtifacts


def _write_manifest(path: Path, entries):
    with open(path, "w", encoding="utf8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_export_personalization_artifacts_creates_expected_files(tmp_path):
    base_audio_dir = tmp_path / "audio"
    base_audio_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    duration_sec = 0.1
    n_samples = int(sr * duration_sec)

    # Create three tiny wave files: one per split
    def _make_wav(rel_path: str):
        full_path = base_audio_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        audio = torch.zeros(n_samples, dtype=torch.float32).numpy()
        # Write simple mono PCM WAV without requiring torchcodec
        sf.write(str(full_path), audio, sr)

    _make_wav("train/spk1_utt1.wav")
    _make_wav("test/spk2_utt1.wav")
    _make_wav("val/spk3_utt1.wav")

    manifest_path = tmp_path / "input_manifest.json"
    entries = [
        {
            "audio_filepath": "train/spk1_utt1.wav",
            "speaker": "spk1",
            "set": "train",
        },
        {
            "audio_filepath": "test/spk2_utt1.wav",
            "speaker": "spk2",
            "set": "test",
        },
        {
            "audio_filepath": "val/spk3_utt1.wav",
            "speaker": "spk3",
            "set": "val",
        },
    ]
    _write_manifest(manifest_path, entries)

    output_dir = tmp_path / "out"

    processor = ExportPersonalizationArtifacts(
        base_audio_dir=str(base_audio_dir),
        output_dir=str(output_dir),
        input_manifest_file=str(manifest_path),
        output_manifest_file=str(output_dir / "primary_manifest.json"),
        sample_rate=sr,
        dtype="int16",
        codec="pcm",
        hdf5_basename="hifitts2_speech",
    )
    processor.process()

    # Check that per-split manifests and HDF5/JSON sidecars exist
    for split in ("train", "test", "val"):
        manifest_split = output_dir / f"manifest_{split}.json"
        assert manifest_split.exists()

        hdf5_path = output_dir / f"hifitts2_speech_{split}.hdf5"
        assert hdf5_path.exists()

        speaker_ids_path = Path(str(hdf5_path) + ".speaker_ids.json")
        speaker_index_path = Path(str(hdf5_path) + ".speaker_index.json")
        assert speaker_ids_path.exists()
        assert speaker_index_path.exists()

        # Inspect HDF5 structure
        with h5.File(str(hdf5_path), "r") as f:
            assert "speech" in f
            grp = f["speech"]
            # Exactly one utterance per split in this test
            assert len(grp.keys()) == 1
            ds_name = next(iter(grp.keys()))
            ds = grp[ds_name]
            assert "n_samples" in ds.attrs
            assert ds.attrs["n_samples"] == n_samples

        # Inspect speaker_ids.json
        with open(speaker_ids_path, "r", encoding="utf8") as fin:
            ids = json.load(fin)
        assert len(ids) == 1
        hdf5_filename = os.path.basename(hdf5_path)
        assert hdf5_filename in ids
        inner = ids[hdf5_filename]
        assert len(inner) == 1
        speaker_id = list(inner.values())[0]
        # speaker should match split-specific speaker
        expected_speaker = {"train": "spk1", "test": "spk2", "val": "spk3"}[split]
        assert speaker_id == expected_speaker

        # Inspect speaker_index.json
        with open(speaker_index_path, "r", encoding="utf8") as fin:
            index_data = json.load(fin)

        assert index_data["total_samples"] == 1
        assert index_data["index_to_speaker"] == [expected_speaker]
        assert index_data["speaker_to_indices"] == {expected_speaker: [0]}

