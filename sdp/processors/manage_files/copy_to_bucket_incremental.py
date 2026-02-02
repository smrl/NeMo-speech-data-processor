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

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CopyToBucketIncremental(BaseParallelProcessor):
    """
    A processor that copies files to a GCS bucket incrementally at the chapter level,
    then deletes local files to free up storage space.
    
    This processor groups manifest entries by chapter directory (based on audio_filepath
    parent directory) and processes one chapter at a time:
    1. Copies all files in a chapter directory to the bucket using gsutil
    2. Deletes local files after successful copy
    3. Updates manifest paths to point to bucket locations
    
    This is designed for large datasets where you don't have enough local storage to
    hold all files before copying to bucket.
    
    Args:
        source_audio_dir (str): Local directory containing audio files to copy.
        bucket_path (str): GCS bucket path (e.g., gs://my-bucket/path/to/data).
        audio_filepath_key (str): Key in manifest entries that contains audio file paths.
            Defaults to "audio_filepath".
        delete_local_files (bool): Whether to delete local files after successful copy.
            Defaults to True.
        update_manifest_paths (bool): Whether to update manifest paths to point to bucket.
            Defaults to True.
        **kwargs: Additional arguments passed to the BaseParallelProcessor.
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.CopyToBucketIncremental
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_bucket.json
              source_audio_dir: ${workspace_dir}/audio_44khz
              bucket_path: gs://my-bucket/hifitts2/audio_44khz
              delete_local_files: true
              update_manifest_paths: true
    """
    
    def __init__(
        self,
        source_audio_dir: str,
        bucket_path: str = None,
        audio_filepath_key: str = "audio_filepath",
        delete_local_files: bool = True,
        update_manifest_paths: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_audio_dir = Path(source_audio_dir)
        self.audio_filepath_key = audio_filepath_key
        self.delete_local_files = delete_local_files
        self.update_manifest_paths = update_manifest_paths
        
        # Handle bucket_path - can be None to skip this processor
        if bucket_path is None:
            self.bucket_path = None
            self.skip = True
        else:
            self.bucket_path = bucket_path.rstrip('/')
            self.skip = False
            # Ensure bucket path starts with gs://
            if not self.bucket_path.startswith('gs://'):
                raise ValueError(f"bucket_path must start with 'gs://', got: {self.bucket_path}")
    
    def prepare(self):
        """Group manifest entries by chapter directory."""
        if self.skip or self.bucket_path is None:
            logger.info("Skipping CopyToBucketIncremental processor (bucket_path not set)")
            return
        
        # Read manifest and group entries by chapter directory
        self.chapter_groups = defaultdict(list)
        
        with open(self.input_manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                if self.audio_filepath_key in entry:
                    audio_path = Path(entry[self.audio_filepath_key])
                    # Get chapter directory (parent of audio file)
                    # Handle both absolute and relative paths
                    if audio_path.is_absolute():
                        try:
                            # Get relative path from source_audio_dir
                            chapter_dir = audio_path.parent.relative_to(self.source_audio_dir)
                        except ValueError:
                            # If not relative to source_audio_dir, use parent as-is
                            chapter_dir = audio_path.parent
                    else:
                        chapter_dir = audio_path.parent
                    self.chapter_groups[chapter_dir].append(entry)
        
        logger.info(f"Grouped {len(self.chapter_groups)} chapters for incremental copying")
        self.chapter_dirs = list(self.chapter_groups.keys())
    
    def read_manifest(self):
        """Return chapter directories to process."""
        if self.skip or self.bucket_path is None:
            # Return empty list to skip processing
            return []
        return self.chapter_dirs
    
    def process_dataset_entry(self, chapter_dir):
        """Process one chapter directory: copy to bucket and optionally delete locally."""
        if self.skip or self.bucket_path is None:
            return []
        
        chapter_dir = Path(chapter_dir)
        bucket_audio_dir = f"{self.bucket_path}/audio"
        
        # Get full path to chapter directory
        chapter_path = self.source_audio_dir / chapter_dir
        
        if not chapter_path.exists():
            logger.warning(f"Chapter directory not found: {chapter_path}, skipping")
            return []
        
        # Copy chapter directory to bucket using gsutil rsync
        # This copies all files in the chapter directory at once
        logger.info(f"Copying chapter {chapter_dir} to bucket...")
        try:
            source_dir = str(chapter_path) + "/"
            bucket_chapter_dir = f"{bucket_audio_dir}/{chapter_dir}"
            result = subprocess.run(
                ["gsutil", "-m", "rsync", "-r", source_dir, bucket_chapter_dir + "/"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Successfully copied chapter {chapter_dir} to bucket")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy chapter {chapter_dir}: {e.stderr}")
            raise RuntimeError(f"gsutil rsync failed for chapter {chapter_dir}: {e.stderr}")
        
        # Delete local chapter directory if requested
        if self.delete_local_files:
            try:
                import shutil
                logger.info(f"Deleting local chapter directory: {chapter_path}")
                shutil.rmtree(chapter_path)
                logger.info(f"Deleted {chapter_path}")
            except Exception as e:
                logger.warning(f"Failed to delete local chapter directory {chapter_path}: {e}")
                # Don't raise - cleanup failure shouldn't fail the whole process
        
        # Return updated entries with bucket paths
        updated_entries = []
        for entry in self.chapter_groups[chapter_dir]:
            if self.update_manifest_paths and self.audio_filepath_key in entry:
                # Update path to point to bucket
                audio_path = Path(entry[self.audio_filepath_key])
                relative_path = audio_path.relative_to(self.source_audio_dir) if audio_path.is_absolute() else audio_path
                bucket_audio_path = f"{bucket_audio_dir}/{relative_path}"
                entry[self.audio_filepath_key] = bucket_audio_path
            updated_entries.append(DataEntry(data=entry))
        
        return updated_entries
