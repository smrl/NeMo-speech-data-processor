# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SQUIM-based quality metrics processor for individual utterance manifests.
Designed for use with HiFi-TTS2 and similar datasets that output per-utterance manifests.
"""

import librosa
import torch
import torchaudio.functional as F
from pathlib import Path
from torchaudio.pipelines import SQUIM_OBJECTIVE

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class SquimQualityMetrics(BaseParallelProcessor):
    """Computes SQUIM quality metrics (PESQ, STOI, SI-SDR) for individual utterances.

    This processor is designed for manifests with individual utterance entries (like HiFi-TTS2),
    unlike TorchSquimObjectiveQualityMetricsProcessor which expects segmented manifests.

    Metrics computed:
        - pesq_squim: Perceptual Evaluation of Speech Quality (1.0-4.5, higher is better)
        - stoi_squim: Short-Time Objective Intelligibility (0-1, higher is better)
        - sisdr_squim: Scale-Invariant Signal-to-Distortion Ratio in dB (higher is better, 15-20 dB is "clean")

    Args:
        audio_dir (str): Root directory where audio files are stored.
        input_audio_key (str): Manifest key containing relative audio path. Defaults to "audio_filepath".
        device (str): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        The same manifest entries with added pesq_squim, stoi_squim, and sisdr_squim fields.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.quality.squim_utterance.SquimQualityMetrics
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_with_quality.json
              audio_dir: ${workspace_dir}/audio_48khz
              max_workers: 4
    """

    def __init__(
        self,
        audio_dir: str,
        input_audio_key: str = "audio_filepath",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_dir = Path(audio_dir)
        self.input_audio_key = input_audio_key

        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        self.device = device
        self.model = SQUIM_OBJECTIVE.get_model()
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    def process_dataset_entry(self, data_entry):
        audio_path = self.audio_dir / data_entry[self.input_audio_key]

        try:
            # Load audio at native sample rate
            audio, sr = librosa.load(path=str(audio_path), sr=None)

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)

            # SQUIM requires 16kHz
            if sr != 16000:
                audio_tensor = F.resample(audio_tensor, sr, 16000)

            # Run inference
            with torch.no_grad():
                if self.device == "cuda":
                    audio_tensor = audio_tensor.cuda()
                stoi, pesq, si_sdr = self.model(audio_tensor)

            # Add metrics to entry
            data_entry["pesq_squim"] = round(pesq.item(), 3)
            data_entry["stoi_squim"] = round(stoi.item(), 3)
            data_entry["sisdr_squim"] = round(si_sdr.item(), 3)

            return [DataEntry(data=data_entry)]

        except Exception as e:
            logger.warning(f"Failed to compute SQUIM metrics for {audio_path}: {e}")
            # Return entry without metrics (will be filtered out if filtering is applied)
            return [DataEntry(data=data_entry)]


class DropLowQuality(BaseParallelProcessor):
    """Drops utterances below quality thresholds based on SQUIM metrics.

    This is a convenience processor that filters based on multiple quality metrics
    in a single pass, which is more efficient than chaining multiple PreserveByValue processors.

    Args:
        min_pesq (float): Minimum PESQ score (1.0-4.5). Defaults to None (no filtering).
        min_stoi (float): Minimum STOI score (0-1). Defaults to None (no filtering).
        min_sisdr (float): Minimum SI-SDR in dB. Defaults to None (no filtering).

    Returns:
        Manifest entries that pass all specified quality thresholds.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.quality.squim_utterance.DropLowQuality
              input_manifest_file: ${workspace_dir}/manifest_with_quality.json
              output_manifest_file: ${workspace_dir}/manifest_filtered.json
              min_pesq: 3.0
              min_stoi: 0.9
              min_sisdr: 15.0
    """

    def __init__(
        self,
        min_pesq: float = None,
        min_stoi: float = None,
        min_sisdr: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_pesq = min_pesq
        self.min_stoi = min_stoi
        self.min_sisdr = min_sisdr

    def process_dataset_entry(self, data_entry):
        # Check PESQ threshold
        if self.min_pesq is not None:
            pesq = data_entry.get("pesq_squim")
            if pesq is None or pesq < self.min_pesq:
                return [DataEntry(data=None, metrics={"dropped_pesq": 1})]

        # Check STOI threshold
        if self.min_stoi is not None:
            stoi = data_entry.get("stoi_squim")
            if stoi is None or stoi < self.min_stoi:
                return [DataEntry(data=None, metrics={"dropped_stoi": 1})]

        # Check SI-SDR threshold
        if self.min_sisdr is not None:
            sisdr = data_entry.get("sisdr_squim")
            if sisdr is None or sisdr < self.min_sisdr:
                return [DataEntry(data=None, metrics={"dropped_sisdr": 1})]

        return [DataEntry(data=data_entry, metrics={"kept": 1})]

    def finalize(self, metrics):
        dropped_pesq = sum(m.get("dropped_pesq", 0) for m in metrics)
        dropped_stoi = sum(m.get("dropped_stoi", 0) for m in metrics)
        dropped_sisdr = sum(m.get("dropped_sisdr", 0) for m in metrics)
        kept = sum(m.get("kept", 0) for m in metrics)

        logger.info(f"Quality filtering results:")
        logger.info(f"  Kept: {kept}")
        if self.min_pesq is not None:
            logger.info(f"  Dropped (PESQ < {self.min_pesq}): {dropped_pesq}")
        if self.min_stoi is not None:
            logger.info(f"  Dropped (STOI < {self.min_stoi}): {dropped_stoi}")
        if self.min_sisdr is not None:
            logger.info(f"  Dropped (SI-SDR < {self.min_sisdr}): {dropped_sisdr}")

        super().finalize(metrics)
