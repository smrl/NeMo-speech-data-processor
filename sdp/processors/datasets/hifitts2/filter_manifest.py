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
Pre-download filtering for HiFi-TTS2 manifests.

HiFi-TTS2 manifests contain precomputed quality metrics that allow filtering
BEFORE downloading audio, saving significant time and bandwidth.

Available precomputed fields in utterance manifest:
  - bandwidth: Estimated bandwidth of the audiobook chapter
  - speaker_count: Number of speakers detected in the utterance
  - wer: ASR word error rate
  - cer: ASR character error rate
  - duration: Utterance duration
  - set: Dataset partition (train/test_seen/dev_seen/test_unseen/dev_unseen)
"""

import json
from pathlib import Path
from typing import List, Optional

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class FilterHiFiTTS2Manifest(BaseProcessor):
    """Pre-filters HiFi-TTS2 manifest BEFORE downloading audio.

    This processor uses the precomputed quality metrics in HiFi-TTS2 manifests
    to filter utterances before downloading, dramatically reducing download time
    and storage requirements.

    Args:
        min_bandwidth (int): Minimum bandwidth in Hz. Default: None (no filtering).
            Recommended: 20000+ for full-band, 16000+ for wideband.
        max_bandwidth (int): Maximum bandwidth in Hz. Default: None (no filtering).
        speaker_count (int): Exact number of speakers required. Default: 1 (single speaker).
            Set to None to disable speaker filtering.
        max_wer (float): Maximum word error rate. Default: None (no filtering).
            Lower is better. Recommended: < 0.1 for clean transcriptions.
        max_cer (float): Maximum character error rate. Default: None (no filtering).
        min_duration (float): Minimum utterance duration in seconds. Default: None.
        max_duration (float): Maximum utterance duration in seconds. Default: None.
        sets (List[str]): Dataset partitions to include. Default: None (all sets).
            Options: ["train", "test_seen", "dev_seen", "test_unseen", "dev_unseen"]

    Returns:
        Filtered manifest containing only utterances that pass all criteria.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.FilterHiFiTTS2Manifest
              input_manifest_file: ${workspace_dir}/manifest_48khz.json
              output_manifest_file: ${workspace_dir}/manifest_filtered_predownload.json
              min_bandwidth: 20000
              speaker_count: 1
              max_wer: 0.1
              min_duration: 1.0
              max_duration: 30.0
              sets: ["train"]
    """

    def __init__(
        self,
        min_bandwidth: Optional[int] = None,
        max_bandwidth: Optional[int] = None,
        speaker_count: Optional[int] = 1,  # Default to single speaker
        max_wer: Optional[float] = None,
        max_cer: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        sets: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.speaker_count = speaker_count
        self.max_wer = max_wer
        self.max_cer = max_cer
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sets = sets

    def process(self):
        input_path = Path(self.input_manifest_file)
        output_path = Path(self.output_manifest_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_count = 0
        kept_count = 0
        filter_stats = {
            "bandwidth": 0,
            "speaker_count": 0,
            "wer": 0,
            "cer": 0,
            "duration": 0,
            "set": 0,
        }

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                total_count += 1
                entry = json.loads(line.strip())

                # Check bandwidth
                if self.min_bandwidth is not None or self.max_bandwidth is not None:
                    bw = entry.get("bandwidth")
                    if bw is not None:
                        if self.min_bandwidth is not None and bw < self.min_bandwidth:
                            filter_stats["bandwidth"] += 1
                            continue
                        if self.max_bandwidth is not None and bw > self.max_bandwidth:
                            filter_stats["bandwidth"] += 1
                            continue

                # Check speaker count
                if self.speaker_count is not None:
                    sc = entry.get("speaker_count")
                    if sc is not None and sc != self.speaker_count:
                        filter_stats["speaker_count"] += 1
                        continue

                # Check WER
                if self.max_wer is not None:
                    wer = entry.get("wer")
                    if wer is not None and wer > self.max_wer:
                        filter_stats["wer"] += 1
                        continue

                # Check CER
                if self.max_cer is not None:
                    cer = entry.get("cer")
                    if cer is not None and cer > self.max_cer:
                        filter_stats["cer"] += 1
                        continue

                # Check duration
                if self.min_duration is not None or self.max_duration is not None:
                    dur = entry.get("duration")
                    if dur is not None:
                        if self.min_duration is not None and dur < self.min_duration:
                            filter_stats["duration"] += 1
                            continue
                        if self.max_duration is not None and dur > self.max_duration:
                            filter_stats["duration"] += 1
                            continue

                # Check set
                if self.sets is not None:
                    s = entry.get("set")
                    if s is not None and s not in self.sets:
                        filter_stats["set"] += 1
                        continue

                # Entry passed all filters
                kept_count += 1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Log statistics
        logger.info(f"Pre-download filtering complete:")
        logger.info(f"  Total entries: {total_count}")
        logger.info(f"  Kept entries: {kept_count} ({100*kept_count/total_count:.1f}%)")
        logger.info(f"  Filtered out:")
        for reason, count in filter_stats.items():
            if count > 0:
                logger.info(f"    - {reason}: {count}")


class FilterHiFiTTS2Chapters(BaseProcessor):
    """Pre-filters HiFi-TTS2 chapter manifest based on utterance filtering.

    After filtering the utterance manifest, this processor filters the chapter
    manifest to only include chapters that have at least one remaining utterance.
    This ensures we only download chapters that contain wanted utterances.

    Args:
        filtered_utterance_manifest (str): Path to the filtered utterance manifest.

    Returns:
        Filtered chapter manifest containing only chapters with remaining utterances.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.FilterHiFiTTS2Chapters
              input_manifest_file: ${workspace_dir}/chapters_48khz.json
              output_manifest_file: ${workspace_dir}/chapters_filtered.json
              filtered_utterance_manifest: ${workspace_dir}/manifest_filtered.json
    """

    def __init__(
        self,
        filtered_utterance_manifest: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filtered_utterance_manifest = Path(filtered_utterance_manifest)

    def process(self):
        # Load the set of audio filepaths we want to keep
        wanted_filepaths = set()
        with open(self.filtered_utterance_manifest, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                wanted_filepaths.add(entry["audio_filepath"])

        logger.info(f"Loaded {len(wanted_filepaths)} wanted utterances from filtered manifest")

        # Filter chapters to only include those with wanted utterances
        input_path = Path(self.input_manifest_file)
        output_path = Path(self.output_manifest_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_chapters = 0
        kept_chapters = 0
        total_utterances_in_kept = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                total_chapters += 1
                chapter = json.loads(line.strip())

                # Filter utterances in this chapter
                filtered_utterances = [
                    utt for utt in chapter.get("utterances", [])
                    if utt["audio_filepath"] in wanted_filepaths
                ]

                if filtered_utterances:
                    # Update chapter with filtered utterances only
                    chapter["utterances"] = filtered_utterances
                    kept_chapters += 1
                    total_utterances_in_kept += len(filtered_utterances)
                    fout.write(json.dumps(chapter, ensure_ascii=False) + "\n")

        logger.info(f"Chapter filtering complete:")
        logger.info(f"  Total chapters: {total_chapters}")
        logger.info(f"  Kept chapters: {kept_chapters}")
        logger.info(f"  Total utterances in kept chapters: {total_utterances_in_kept}")
