## HiFiTTS‑2 → DeepFilterNet Personalization Artifacts

This document describes how to take a filtered HiFiTTS‑2 manifest and convert it into
DeepFilterNet‑style personalization artifacts (audio HDF5 + speaker sidecars) using
two SDP processors:

- `FilterAndSpeakerSplitManifest`
- `ExportPersonalizationArtifacts`

It assumes:

- You already ran a HiFiTTS‑2 config that downloads audio and produces a filtered
  NeMo manifest (e.g. quality + duration filtering).
- Your audio files are stored under a known root directory (often a bucket mount).

---

## 1. Processors

### 1.1 `FilterAndSpeakerSplitManifest`

**Module**: `sdp.processors.FilterAndSpeakerSplitManifest`

**Purpose**:

- Drop short utterances (`duration < min_duration`).
- Assign each *speaker* to exactly one of the splits:
  - `train`, `test`, `val`
- Update each surviving manifest entry with a `set` field.

**Key properties**:

- **Speaker‑grouped** split: the same speaker will not appear in more than one split.
- Approximate target ratios (by utterance count) via greedy assignment.
- Deterministic with a fixed `seed`.

**Config arguments**:

- `input_manifest_file` (str): path to existing NeMo manifest (JSONL).
- `output_manifest_file` (str): path to the new split manifest (JSONL).
- `min_duration` (float): minimum allowed `duration` in seconds (default: `3.0`).
- `train_ratio` (float): fraction for train split (default: `0.75`).
- `test_ratio` (float): fraction for test split (default: `0.15`).
- `val_ratio` (float): fraction for validation split (default: `0.10`).
- `speaker_key` (str): manifest key for speaker ID (default: `"speaker"`).
- `set_key` (str): manifest key to write split name into (default: `"set"`).
- `seed` (int): RNG seed for deterministic speaker shuffling (default: `0`).

**Input manifest requirements**:

- JSONL NeMo manifest with at least:

  - `audio_filepath`: path relative to your audio root
  - `duration`: float (seconds)
  - `speaker`: string or int

**Output manifest**:

- Same entries as input but:

  - Entries with `duration < min_duration` removed.
  - Remaining entries have `set` ∈ `{train, test, val}`.

Example config snippet:

```yaml
- _target_: sdp.processors.FilterAndSpeakerSplitManifest
  input_manifest_file: ${workspace_dir}/manifest_filtered_48khz.json
  output_manifest_file: ${workspace_dir}/manifest_split_personalization.json
  min_duration: 3.0
  train_ratio: 0.75
  test_ratio: 0.15
  val_ratio: 0.10
  speaker_key: speaker
  set_key: set
  seed: 0
```

---

### 1.2 `ExportPersonalizationArtifacts`

**Module**: `sdp.processors.ExportPersonalizationArtifacts`

**Purpose**:

From a single, split‑annotated manifest, create for each split (`train`, `test`, `val`):

- A NeMo manifest (JSONL).
- A speech HDF5 file with PCM samples.
- `speaker_ids.json` (HDF5 key → logical speaker ID).
- `speaker_index.json` (speaker ↔ sample index mapping).

This matches the artifact structure described in
`REFERENCE/personalization_data_pipeline.md` and the PTDB/VCTK/EARS examples.

**Config arguments**:

- `input_manifest_file` (str): path to split manifest (JSONL) with `set`.
- `output_manifest_file` (str): a convenience copy of `manifest_train.json`.
- `base_audio_dir` (str): root directory where `audio_filepath` is relative.
  - Example: bucket mount: `/home/user/NeMo-speech-data-processor/dataset/audio_48khz`.
- `output_dir` (str): directory where all HDF5 and JSON sidecars are written.
- `audio_key` (str): key for relative audio path (default: `audio_filepath`).
- `speaker_key` (str): key for speaker ID (default: `speaker`).
- `set_key` (str): key for split label (default: `set`).
- `splits` (list[str], optional): which splits to export (default: `["train", "test", "val"]`).
- `sample_rate` (int): target sample rate (default: `48000`).
- `dtype` (`"int16"` or `"float32"`): storage dtype for audio (default: `"int16"`).
- `codec` (str): currently only `"pcm"` is supported.
- `hdf5_basename` (str): base name for HDF5 files (default: `"speech"`).

**HDF5 layout**:

- File per split:

  - `<output_dir>/<hdf5_basename>_<split>.hdf5`

- Structure:

  - Root attributes:
    - `sr`, `dtype`, `codec`
  - Group `/speech`:
    - One dataset per utterance:
      - **Key**:
        - `rel_path = os.path.relpath(audio_path, base_audio_dir)`
        - `hdf5_key = rel_path.replace("/", "_")`
      - **Value**:
        - 1D PCM samples in `dtype`.
      - **Attribute**:
        - `n_samples` (int)

**Sidecar JSONs**:

- `speaker_ids.json`:

  ```json
  {
    "<hdf5_filename>": {
      "<hdf5_key>": "<speaker_id>",
      "....": "...."
    }
  }
  ```

- `speaker_index.json`:

  ```json
  {
    "speaker_to_indices": {
      "<speaker_id>": [0, 5, 7, ...],
      "...": [...]
    },
    "index_to_speaker": [
      "<speaker_id_or_null>",
      "...",
      null
    ],
    "total_samples": <int>
  }
  ```

**Example config snippet**:

```yaml
- _target_: sdp.processors.ExportPersonalizationArtifacts
  input_manifest_file: ${workspace_dir}/manifest_split_personalization.json
  output_manifest_file: ${workspace_dir}/manifest_train_personalization.json
  base_audio_dir: ${audio_source_dir}
  output_dir: ${workspace_dir}/dfn_personalization
  audio_key: audio_filepath
  speaker_key: speaker
  set_key: set
  sample_rate: 48000
  dtype: int16
  codec: pcm
  hdf5_basename: hifitts2_speech
  splits:
    - train
    - test
    - val
```

---

## 2. End‑to‑end HiFiTTS‑2 → DFN pipeline

### 2.1 Prerequisites

1. **HiFiTTS‑2 download + filtering**  
   Run one of the HiFiTTS‑2 configs (e.g. `config_48khz_deepfilternet.yaml`) to:

   - Download LibriVox audio at 48 kHz.
   - Remove failed chapters.
   - Optionally filter by quality and duration.

   This should leave you with a manifest that looks roughly like:

   ```json
   {
     "audio_filepath": "16/1525/16_1525..._44.flac",
     "speaker": "16",
     "duration": 15.64,
     "bandwidth": 16537,
     "speaker_count": 1,
     "text": "...",
     "normalized_text": "..."
   }
   ```

2. **Audio location**  
   Know the root directory that `audio_filepath` is relative to, e.g.:

   ```bash
   /home/user/NeMo-speech-data-processor/dataset/audio_48khz
   ```

3. **Python deps**  
   Make sure your environment has:

   - `torchaudio`, `torch`, `h5py`, `soundfile`, and `torchcodec` (for recent torchaudio).

---

### 2.2 Example config file

A ready‑to‑use config is provided at:

- `dataset_configs/english/hifitts2/config_48khz_dfn_personalization.yaml`

It assumes:

- You already have a filtered manifest:
  - `${workspace_dir}/manifest_filtered_48khz.json`
- Audio is under a separate root:
  - `${audio_source_dir}` (e.g. bucket mount)

The config:

1. Runs `FilterAndSpeakerSplitManifest` on your filtered manifest.
2. Runs `ExportPersonalizationArtifacts` to write:

   - `manifest_train.json`, `manifest_test.json`, `manifest_val.json`
   - `hifitts2_speech_train.hdf5` (+ sidecars)
   - `hifitts2_speech_test.hdf5` (+ sidecars)
   - `hifitts2_speech_val.hdf5` (+ sidecars)

---

## 3. How to run

From the repo root:

```bash
cd /home/user/360/NeMo-speech-data-processor

uv run python main.py \
  dataset_configs/english/hifitts2/config_48khz_dfn_personalization.yaml \
  workspace_dir=/home/user/NeMo-speech-data-processor/workspace \
  filtered_manifest=/home/user/NeMo-speech-data-processor/workspace/manifest_filtered_48khz.json \
  audio_source_dir=/home/user/NeMo-speech-data-processor/dataset/audio_48khz \
  hdf5_output_dir=/home/user/NeMo-speech-data-processor/workspace/dfn_personalization
```

Adjust the paths to match your environment and bucket mount.

After this finishes, the `hdf5_output_dir` will contain the DFN‑ready artifacts
for train, test, and validation splits.

