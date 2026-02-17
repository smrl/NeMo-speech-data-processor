# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import collections
import os
import re
from array import array
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Dict, List, Optional, Tuple
from tempfile import NamedTemporaryFile
import tempfile
import shutil
import requests
import wget
import tarfile
from glob import glob
import yaml

import soundfile
import torchaudio
from docx import Document
from tqdm import tqdm
import json
import librosa
import numpy as np
from pathlib import Path
import h5py as h5

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import ffmpeg_convert, load_manifest, save_manifest
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces
from sdp.utils.get_diff import get_diff_with_subs_grouped
from sdp.utils.metrics_computation import get_wer
from sdp.utils.apply_operators import evaluate_expression


class GetAudioDuration(BaseParallelProcessor):
    """
    Processor that computes the duration of the file in ``audio_filepath_key`` (using soundfile)
    and saves the duration in ``duration_key``. If there is an error computing the duration,
    the value at ``duration_key`` will be updated with the value -1.0.

    Args:
        audio_filepath_key (str): Key to get path to wav file.
        duration_key (str): Key to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_key
    """

    def __init__(
        self,
        audio_filepath_key: str,
        duration_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.duration_key = duration_key

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.audio_filepath_key]
        try:
            data, samplerate = soundfile.read(audio_filepath)
            data_entry[self.duration_key] = data.shape[0] / samplerate
        except Exception as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.duration_key] = -1.0
        return [DataEntry(data=data_entry)]


class ReadTxtLines(BaseParallelProcessor):
    """
    The text file specified in source_filepath will be read, and each line in it will be added as a line in the output manifest,
    saved in the field text_key.

    Args:
        input_file_key (str): The key in the manifest containing the input txt file path .
        text_key (str): The key to store the read text lines in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        input_file_key: str,
        text_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_file_key = input_file_key
        self.text_key = text_key

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_file_key]
        data_list = []
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = data_entry.copy()
                    data[self.text_key] = line
                    data_list.append(DataEntry(data=data))
        return data_list


class CountNumWords(BaseParallelProcessor):
    """
    A processor that counts the number of words in the `text_key` field of each dataset entry and stores the result in `num_words_key`.

    Before counting, the text is optionally cleaned using a custom `alphabet`:
    - If `alphabet` is provided, all characters not in the alphabet are replaced with whitespace.
    - Consecutive whitespace characters are collapsed into a single space.
    - The number of resulting space-separated tokens is counted as the number of words.

    Args:
        text_key (str): The key in the input data entry containing the text to be analyzed.
        num_words_key (str): The key under which the word count will be stored in the output entry. Defaults to "num_words".
        alphabet (str, optional): A string of allowed characters (e.g., lowercase letters). All characters not in this set will be replaced with whitespace before counting. If not provided, no filtering is applied.
        **kwargs: Additional arguments passed to the BaseParallelProcessor.

    Returns:
        A manifest where each entry is the original data entry with an added field `num_words_key` (default: `"num_words"`),
        indicating the number of words in the `text_key` field.
    """

    def __init__(
        self,
        text_key: str,
        num_words_key: str = "num_words",
        alphabet: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.num_words_key = num_words_key
        self.pattern = None
        if alphabet:
            self.pattern = re.compile("[^" + alphabet + "]")

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_key]
        cleaned_string = text
        if self.pattern:
            cleaned_string = self.pattern.sub("", cleaned_string).strip()
        cleaned_string = re.sub("\\s+", " ", cleaned_string).strip()
        words = cleaned_string.split()
        num_words = len(words)
        data_entry[self.num_words_key] = num_words
        return [DataEntry(data=data_entry)]


class SplitLineBySentence(BaseParallelProcessor):
    """
    Processor for splitting lines of text into sentences based on a specified pattern.
    One line containing N sentences will be transformed into N lines containing one sentence.

    Args:
        text_key (str): The field containing the text lines in the dataset.
        end_pattern (str): The regular expression pattern to identify sentence boundaries.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.
    """

    def __init__(
        self,
        text_key: str,
        end_pattern: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pattern = re.compile(end_pattern)

    def process_dataset_entry(self, data_entry):
        line = data_entry[self.text_key]
        data_list = []
        start = 0
        ends = [m.start() for m in self.pattern.finditer(line)]
        if ends:
            for end in ends:
                sent = line[start : end + 1].strip()
                # if sent and sent[0].isupper():
                data = data_entry.copy()
                data[self.text_key] = sent
                data_list.append(DataEntry(data=data))
                start = end + 1
            if start < len(line):
                pass
        else:
            data = data_entry.copy()
            data[self.text_key] = line.strip()
            data_list.append(DataEntry(data=data))
        return data_list


class InsIfASRInsertion(BaseParallelProcessor):
    """Processor that adds substrings to transcription if they are present in ASR predictions.

    Will insert substrings into ``data[self.text_key]`` if it is
    present at that location in ``data[self.pred_text_key]``.
    It is useful if words are systematically missing from ground truth
    transcriptions.

    Args:
        insert_words (list[str]): list of strings that will be inserted
            into ``data[self.text_key]`` if there is an insertion (containing
            only that string) in ``data[self.pred_text_key]``.
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

            .. note::
                Because this processor looks for an exact match in the insertion,
                we recommend including variations with different spaces in
                ``insert_words``, e.g. ``[' nemo', 'nemo ', ' nemo ']``.

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        insert_words: List[str],
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.insert_words = insert_words
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    def process_dataset_entry(self, data_entry) -> List:
        insert_word_counter = collections.defaultdict(int)
        for insert_word in self.insert_words:
            if not insert_word in data_entry[self.pred_text_key]:
                break
            orig_words, pred_words = (
                data_entry[self.text_key],
                data_entry[self.pred_text_key],
            )
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text
                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        if diff_entry[1] == insert_word:
                            new_sent += insert_word
                            insert_word_counter[insert_word] += 1

                    elif isinstance(diff_entry, tuple):  # i.e. diff is a substitution
                        new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = " ".join(new_sent.split())  # remove any extra spaces
                data_entry[self.text_key] = new_sent

        return [DataEntry(data=data_entry, metrics=insert_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Num of words that were inserted")
        for word, count in total_counter.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)


class SubIfASRSubstitution(BaseParallelProcessor):
    """Processor that substitutes substrings to transcription if they are present in ASR predictions.

    Will convert a substring in ``data[self.text_key]`` to a
    substring in ``data[self.pred_text_key]`` if both are located in the
    same place (ie are part of a 'substitution' operation) and if the substrings
    correspond to key-value pairs in ``sub_words``.
    This is useful if words are systematically incorrect in ground truth
    transcriptions.

    Before starting to look for substitution, this processor adds spaces at the beginning and end of
    ``data[self.text_key]`` and ``data[self.pred_text_key]``, to ensure that an argument like
    ``sub_words = {"nmo ": "nemo "}`` would cause a substitution to be made even if the original
    ``data[self.text_key]`` ends with ``"nmo"`` and ``data[self.pred_text_key]`` ends with ``"nemo"``.

    Args:
        sub_words (dict): dictionary where a key is a string that might be in
            ``data[self.text_key]`` and the value is the string that might
            be in ``data[self.pred_text_key]``. If both are located in the same
            place (i.e. are part of a 'substitution' operation)
            then the key string will be converted to the value string
            in ``data[self.text_key]``.
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

            .. note::
                This processor looks for exact string matches of substitutions,
                so you may need to be careful with spaces in ``sub_words``. E.g.
                it is recommended to do ``sub_words = {"nmo ": "nemo "}``
                instead of ``sub_words = {"nmo" : "nemo"}``.

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        sub_words: Dict,
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_words = sub_words
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    def process_dataset_entry(self, data_entry) -> List:
        sub_word_counter = collections.defaultdict(int)
        data_entry[self.text_key] = add_start_end_spaces(data_entry[self.text_key])
        data_entry[self.pred_text_key] = add_start_end_spaces(data_entry[self.pred_text_key])
        for original_word, new_word in self.sub_words.items():
            if not original_word in data_entry[self.text_key]:
                break
            orig_words, pred_words = (
                data_entry[self.text_key],
                data_entry[self.pred_text_key],
            )
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text
                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        # don't make changes
                        pass

                    elif isinstance(diff_entry, tuple):  # substitution
                        if diff_entry[0][1] == original_word and diff_entry[1][1] == new_word:
                            # ie. substitution is one we want to use to change the original text
                            new_sent += new_word
                            sub_word_counter[original_word] += 1

                        else:
                            # ie. substitution is one we want to ignore
                            new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = add_start_end_spaces(new_sent)
                data_entry[self.text_key] = new_sent

        data_entry[self.text_key] = remove_extra_spaces(data_entry[self.text_key])
        data_entry[self.pred_text_key] = remove_extra_spaces(data_entry[self.pred_text_key])

        return [DataEntry(data=data_entry, metrics=sub_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Num of words that were substituted")
        for word, count in total_counter.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)


# TODO: replace with generic regex


class SubMakeLowercase(BaseParallelProcessor):
    """Processor to convert text to lowercase.

    text_key (str): a string indicating which key of the data entries
        should be used to find the utterance transcript. Defaults to "text".

    Returns:
        The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        data_entry[self.text_key] = data_entry[self.text_key].lower()
        return [DataEntry(data=data_entry)]

    def finalize(self, metrics):
        logger.info("Made all letters lowercase")
        super().finalize(metrics)


class SubRegex(BaseParallelProcessor):
    """
    Applies a sequence of regex substitutions to the specified text field in each data entry.

    This processor performs regex-based substitutions as defined in either a provided list of
    regex parameter dictionaries or a YAML configuration file. Each substitution is applied in
    the order specified.

    Before substitutions are applied, a space is temporarily added to the beginning and end of the text
    to improve regex match consistency. After all substitutions, leading/trailing spaces and repeated
    spaces are removed.

    Args:
        regex_params_list (List[Dict], optional): A list of dictionaries specifying the regex substitutions.
            Each dictionary must include::

                - "pattern": A regex pattern to match.
                - "repl": A replacement string.
                - "count" (optional): Maximum number of replacements to make. Defaults to 0 (replace all).

        regex_params_yaml (str, optional): Path to a YAML file that defines the same list of dictionaries
            as `regex_params_list`. Either `regex_params_list` or `regex_params_yaml` must be provided.
            If both are provided, `regex_params_yaml` takes precedence.

        text_key (str): The key in each data entry whose value will be modified. Defaults to "text".

        **kwargs: Additional arguments passed to the BaseParallelProcessor.

    Example YAML format for `regex_params_yaml`:
        ```
        # regex_params.yaml
        - {"pattern": "♩", "repl": " "}
        - {"pattern": "♭", "repl": " "}
        - {"pattern": "\\|", "repl": " "}
        - {"pattern": ":", "repl": " "}
        - {"pattern": "-", "repl": " "}
        - {"pattern": "[^ €₽₴$£%?!',.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЬЮЯабвгдежзийклмнопрстуфхцчшщъьюя]", "repl": ""}
        - {"pattern": "\\s+\\.", "repl": "."}
        - {"pattern": "\\?+", "repl": "?"}
        - {"pattern": "\\.+", "repl": "."}
        ```

    Returns:
        The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        regex_params_list: List[Dict] = None,
        regex_params_yaml: str = None,
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not regex_params_list and not regex_params_yaml:
            raise ValueError(f'One of `regex_params_list` or `regex_params_yaml` should be provided.')
        
        self.regex_params_list = regex_params_list
        if regex_params_yaml:
            with open(regex_params_yaml, 'r') as regex_params_file: 
                self.regex_params_list = yaml.safe_load(regex_params_file)

        self.text_key = text_key

        # verify all dicts in regex_params_list have "pattern" and "repl" keys
        for regex_params_dict in self.regex_params_list:
            if not "pattern" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'pattern' in all entries of `regex_params_list`: {self.regex_params_list}"
                )
            if not "repl" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'repl' in all entries of `regex_params_list`: {self.regex_params_list}"
                )

    def process_dataset_entry(self, data_entry) -> List:
        """Replaces each found regex match with a given string."""
        replace_word_counter = collections.defaultdict(int)

        text_in = data_entry[self.text_key]

        text_in = add_start_end_spaces(text_in)
        for regex_params in self.regex_params_list:
            text_out = re.sub(
                pattern=regex_params["pattern"],
                repl=regex_params["repl"],
                string=text_in,
                # note: this count param is the maximum number of pattern occurrences to be replaced.
                count=regex_params.get("count", 0),
            )

            if text_in != text_out:
                replace_word_counter[regex_params["pattern"]] += 1
            text_in = text_out

        text_out = remove_extra_spaces(text_out)

        data_entry[self.text_key] = text_out

        return [DataEntry(data=data_entry, metrics=replace_word_counter)]

    def finalize(self, metrics):
        """Reports how many substitutions were made for each pattern."""
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Number of utterances which applied substitutions for the following patterns:")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))
        for word, count in total_counter_sorted.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)


class NormalizeText(BaseParallelProcessor):
    """This processor applies text normalization (TN) to the text. I.e. converts text from written form into its verbalized form.
    E.g., "$123" is converted to "one hundred and twenty-three dollars."

    Args:
        input_text_key (str): the text field that will be the input to the Normalizer. Defaults to: text.
        input_language (str): language specifying the text normalization rules in ISO 639 Set 1 format. E.g., "en", "es", "it", etc.
            Defaults to: English.
        input_case (str): input text capitalization, set to `cased` if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
            Defaults to: cased.
        output_text_key (str): the text field that will be the output from the Normalizer.
            Defaults to: text.

    Returns:
        This processor normalizes the text in the `input_text_key` field and saves the normalized text in `output_text_key` field.

    Raises:
        `NotImplementedError`: when TN is not implemented for the requested language.
    """

    def __init__(
        self,
        input_text_key: str = "text",
        input_language: str = "en",
        input_case: str = "cased",
        output_text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.input_case = input_case
        self.input_language = input_language

    def prepare(self):
        from nemo_text_processing.text_normalization.normalize import Normalizer
        try:
            self.normalizer = Normalizer(input_case=self.input_case, lang=self.input_language)
        except NotImplementedError as e:
            logger.error("Failed to run text normalization: %s", repr(e))

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_text_key] = self.normalizer.normalize(data_entry[self.input_text_key])
        return [DataEntry(data=data_entry)]


class InverseNormalizeText(BaseParallelProcessor):
    """This processor applies inverse text normalization (ITN) to the text. I.e. transforms spoken forms of numbers, dates, etc into their written equivalents.
    E.g., "one hundred and twenty-three dollars." is converted to "$123".

    Args:
        input_text_key (str): the text field that will be the input to the InverseNormalizer. Defaults to: text.
        input_language (str): language specifying the text normalization rules in ISO 639 Set 1 format. E.g., "en", "es", "it", etc.
            Defaults to: English.
        input_case (str): input text capitalization, set to `cased` if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
            Defaults to: cased.
        output_text_key (str): the text field that will be the output from the InverseNormalizer.
            Defaults to: text.

    Returns:
        This processor inverse normalizes the text in the `input_text_key` field and saves the inverse normalized text in `output_text_key` field.

    Raises:
        `NotImplementedError`: when ITN is not implemented for the requested language.
    """

    def __init__(
        self,
        input_text_key: str = "text",
        input_language: str = "en",
        input_case: str = "cased",
        output_text_key: str = "text",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.input_case = input_case
        self.input_language = input_language
        self.verbose = verbose

    def prepare(self):
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
        try:
            self.inverse_normalizer = InverseNormalizer(input_case=self.input_case, lang=self.input_language)
        except NotImplementedError as e:
            logger.error("Failed to run text inverse normalization: %s", repr(e))

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_text_key] = self.inverse_normalizer.inverse_normalize(
            data_entry[self.input_text_key], verbose=self.verbose
        )
        return [DataEntry(data=data_entry)]


class CopyManifestData(BaseParallelProcessor):
    """This processor copies files specified in the manifest to a new location.

    It is useful for creating a consolidated dataset by gathering files from different sources
    into a single directory.

    Args:
        copy_path (str): The destination directory where files will be copied.
        source_filepath (str): The key in the manifest that contains the path to 
            the file to be copied. Default: "audio_path".

    Returns:
        The same data as in the input manifest, but the files referenced in the manifest
        will have been copied to the specified destination directory.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.modify_manifest.data_to_data.CopyManifestData
              input_manifest_file: ${workspace_dir}/dataset.json
              output_manifest_file: ${workspace_dir}/dataset_copied.json
              copy_path: ${workspace_dir}/consolidated_data
              source_filepath: "audio_filepath"
    """
    def __init__(
        self,
        copy_path: str,
        source_filepath: str = "audio_path",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = source_filepath
        self.copy_path = copy_path

    def prepare(self):
        os.makedirs(self.copy_path, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_field]

        dest_file_path = os.path.join(self.copy_path, os.path.basename(fname))
        shutil.copy(fname, dest_file_path)
        data_entry[self.input_field] = dest_file_path

        return [DataEntry(data=data_entry)]


class ReadDocxLines(BaseParallelProcessor):
    """
    Processor for reading text lines from a docx file and updating the manifest.

    Args:
        source_filepath (str): The field containing the file path in the manifest.
        text_key (str): The field to store the read text lines in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        source_filepath: str,
        text_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = source_filepath
        self.output_field = text_key

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_field]

        # Skip hidden files and directories (e.g., .DS_Store, ._filename)
        if os.path.basename(fname).startswith('.'):
            logger.warning(f"Skipping hidden file: {fname}")
            return []

        data_list = []

        try:
            doc = Document(fname)
            for para in doc.paragraphs:
                line = para.text.strip()
                if line:
                    data = data_entry.copy()
                    data[self.output_field] = line
                    data_list.append(DataEntry(data=data))
        except Exception as e:
            logger.error(f"Error reading document {fname}: {e}")

        return data_list


class ExtractFromBrackets(BaseParallelProcessor):
    """
    A class for extracting text contained within specified bracket types from strings,
    handling nested brackets.

    Example Input:
        data_entry = {
            "text": "This is a [test] string with [multiple [nested] brackets]."
        }

    Example Output:
        [
            {
                "text": "test"
            },
            {
                "text": "multiple [nested] brackets"
            }
        ]

    Explanation:
        - It extracts "test" from the first occurrence of brackets.
        - It extracts "multiple [nested] brackets" from the second occurrence, handling nested brackets correctly.

    Attributes:
        brackets (List[str]): A list where each element is a pair of strings representing
                              the opening and closing brackets.
        text_key (str): The key in the input data from which to extract text, defaults to "text".
    """

    def __init__(
        self,
        brackets: List[str],
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brackets = brackets
        self.text_key = text_key

    def extract_text_within_brackets(self, text, brackets):
        """
        Extracts text within the specified brackets, including handling nested brackets.

        Args:
            text (str): The string from which to extract text.
            brackets (tuple[str, str]): A tuple containing the opening and closing bracket.

        Returns:
            List[str]: A list of strings, each representing a segment of text found within
                       the outermost brackets, including any nested brackets content.
        """
        open_bracket, close_bracket = brackets
        depth = 0
        buffer = ""
        sentences = []

        for char in text:
            if char == open_bracket:
                if depth > 0:
                    buffer += char  # Add to buffer if already inside brackets
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:  # Exiting outermost brackets
                    if buffer:
                        sentences.append(buffer)
                        buffer = ""  # Reset buffer for next possible extraction
                elif depth > 0:
                    buffer += char  # Still inside nested brackets, continue adding
            elif depth > 0:
                buffer += char  # Add characters inside brackets to buffer

        return sentences

    def process_dataset_entry(self, data_entry) -> List:
        data: list[dict] = []
        sentences = []
        text_in = data_entry[self.text_key]

        for bracket in self.brackets:
            sentences.extend(self.extract_text_within_brackets(text_in, bracket))

        for sentence in sentences:
            new_entry = data_entry.copy()
            new_entry[self.text_key] = sentence
            # new_entry["ORIGINAL TEXT"] = text_in  # for testing
            data.append(new_entry)

        data_list = []
        for data_point in data:
            data_list.append(DataEntry(data=data_point))

        return data_list


class GetWER(BaseParallelProcessor):
    """This processor calculates Word Error Rate (WER) between predicted text and ground truth text.

    It computes the WER for each entry in the manifest and adds the result as a new field.
    
    Args:
        text_key (str): Key for the ground truth text field in the manifest. Default: "text".
        pred_text_key (str): Key for the predicted text field in the manifest. Default: "pred_text".
    
    Returns:
        The same data as in the input manifest with an additional 'wer' field containing 
        the calculated Word Error Rate between the specified text fields.
    """
    def __init__(
        self,
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    def process_dataset_entry(self, data_entry) -> List:
        data_entry['wer'] = get_wer(data_entry[self.text_key], data_entry[self.pred_text_key])
        return [DataEntry(data=data_entry)]


class MakeSentence(BaseParallelProcessor):
    """This processor formats text strings into proper sentences.

    It capitalizes the first character of the text (if enabled) and appends
    an end symbol if the text does not already end with punctuation.

    Args:
        text_key (str): The key in the manifest containing the text to be processed.
            Default: "text".
        end_symbol (str): The punctuation symbol to add at the end of the text if it
            doesn't already have one. Default: ":".
        make_uppercase (bool): Whether to capitalize the first character of the text.
            Default: True.

    Returns:
        The same data as in the input manifest with the text field modified to have
        proper sentence formatting.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.modify_manifest.data_to_data.MakeSentence
              input_manifest_file: ${workspace_dir}/dataset.json
              output_manifest_file: ${workspace_dir}/dataset_formatted.json
              text_key: "transcript"
              end_symbol: "."
              make_uppercase: true
    """
    def __init__(
        self,
        text_key: str = "text",
        end_symbol: str = ":",
        make_uppercase: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.make_uppercase = make_uppercase
        self.text_key = text_key
        self.end_symbol = end_symbol

    def process_dataset_entry(self, data_entry) -> List:
        if self.make_uppercase:
            data_entry[self.text_key] = data_entry[self.text_key][0].upper() + data_entry[self.text_key][1:]

        # Append end_symbol only if the text doesn't end with punctuation
        if data_entry[self.text_key][-1].isalpha():
            data_entry[self.text_key] += self.end_symbol
        return [DataEntry(data=data_entry)]


class ASRFileCheck(BaseProcessor):
    """This processor validates audio files in the manifest and identifies corrupted files.

    It attempts to load each audio file using the torchaudio library and moves corrupted
    files to a specified directory.

    Args:
        audio_filepath_key (str): The key in the manifest that contains the path to
            the audio file. Default: "audio_filepath".
        corrupted_audio_dir (str): The directory where corrupted audio files will be moved.
        workspace_dir (str, optional): The base directory for resolving relative paths.
            Default: None.

    Returns:
        A manifest with corrupted audio files removed.

    """
    def __init__(self, audio_filepath_key: str = "audio_filepath", corrupted_audio_dir: str = None, workspace_dir: str = None, **kwargs):
        """
        Constructs the necessary attributes for the ASRFileCheck class.

        Parameters:
        ----------
        audio_filepath_key : str, optional
            The key in the manifest entries used to retrieve the path to the audio file. Defaults to 'audio_filepath'.
        corrupted_audio_dir : str
            The directory where corrupted audio files will be moved. This is required.
        workspace_dir : str, optional
            The base directory where audio files are stored. If provided, audio file paths will be resolved
            relative to this directory. Defaults to None.
        """
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        
        if corrupted_audio_dir is None:
            raise ValueError("corrupted_audio_dir parameter is required. Please specify a directory to move corrupted files.")
        
        self.corrupted_audio_dir = corrupted_audio_dir
        self.workspace_dir = workspace_dir
        self.failed_files = []

    def process(self):
        """
        Check each file listed in the manifest to ensure it can be loaded with torchaudio.

        This method reads through the manifest file, attempts to load each audio file using torchaudio,
        and moves corrupted files. A new manifest file is created with only the valid entries.
        
        Specific errors handled:
        - FileNotFoundError: File doesn't exist
        - RuntimeError: File format issues or codec problems
        - Other exceptions: General issues with file loading
        """
        from sdp.logging import logger
        
        # Debug print to show workspace_dir
        logger.info(f"ASRFileCheck workspace_dir: {self.workspace_dir}")
        
        with open(self.input_manifest_file, 'r') as f:
            lines = f.readlines()

        entries = []
        total_lines = len(lines)

        # Ensure the corrupted files directory exists
        os.makedirs(self.corrupted_audio_dir, exist_ok=True)

        for idx in tqdm(range(total_lines), desc="Checking Audio Files"):
            line = lines[idx]
            entry = json.loads(line)
            audio_path = entry[self.audio_filepath_key]
            
            # Debug print first file path
            if idx == 0:
                logger.info(f"First audio_path from manifest: {audio_path}")
            
            # If workspace_dir is provided, join it with audio_path to get absolute path
            if self.workspace_dir is not None:
                full_audio_path = os.path.join(self.workspace_dir, audio_path)
            else:
                full_audio_path = audio_path
            
            # Debug print first full path
            if idx == 0:
                logger.info(f"First full_audio_path: {full_audio_path}")
                logger.info(f"Path exists: {os.path.exists(full_audio_path)}")
            
            try:
                # Attempt to load the audio file to check if it is corrupted
                torchaudio.load(full_audio_path)
                entries.append(entry)  # File is good, append to entries list
            except FileNotFoundError:
                logger.warning(f"File not found: {full_audio_path}")
                self.failed_files.append(audio_path)
            except RuntimeError as e:
                logger.warning(f"Audio format error in {audio_path}: {e}")
                self.failed_files.append(audio_path)
                
                # Move the corrupted audio file
                if os.path.exists(full_audio_path):
                    dest_path = os.path.join(self.corrupted_audio_dir, os.path.basename(audio_path))
                    os.rename(full_audio_path, dest_path)
                    logger.info(f"Moved corrupted file to: {dest_path}")
            except Exception as e:
                logger.warning(f"Unknown error loading {audio_path}: {e}")
                self.failed_files.append(audio_path)
                
                # Move the corrupted audio file
                if os.path.exists(full_audio_path):
                    dest_path = os.path.join(self.corrupted_audio_dir, os.path.basename(audio_path))
                    os.rename(full_audio_path, dest_path)
                    logger.info(f"Moved corrupted file to: {dest_path}")

        # Output non-corrupted entries to a new manifest file
        with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
            for entry in entries:
                json.dump(entry, f_out, ensure_ascii=False)
                f_out.write("\n")

        if self.failed_files:
            logger.warning(f"Failed to process {len(self.failed_files)} files.")
            logger.debug(f"Failed files: {self.failed_files}")


class ListToEntries(BaseParallelProcessor):
    """
    A dataset processor that transforms a single entry containing a list of items into multiple entries,
    one for each item in the list.

    This is useful when a manifest field (e.g., "segments") contains a list of sub-entries, and you want
    to flatten these into individual records for further processing.

    Args:
        field_with_list (str): The name of the field in the input entry that contains a list.
        output_field (str, optional): The name of the output field to assign to items in the list
            if they are not dictionaries. Required if the list contains primitive types (e.g., strings).
        **kwargs: Additional arguments passed to the BaseParallelProcessor.

    Raises:
        TypeError: If the specified list field is not of type list.
        ValueError: If the list items are not dictionaries and `output_field` is not provided.
    
    Returns:
        A manifest where each entry corresponds to one item in the original list from the input entry. 
        This effectively transforms a single input entry containing a list of items into multiple standalone 
        entries, each suitable for further dataset processing.

    .. admonition:: Example 1 (list of dicts)
        
        .. code-block:: yaml
    
            - _target_: sdp.processors.ListToEntries
              input_manifest_file: ${workspace_dir}/input_manifest.json
              output_manifest_file: ${workspace_dir}/output_manifest.json
              field_with_list: "segments"
                
        Input::
 
            {
                "audio_filepath": "sample.wav",
                "segments": [
                    {"start": 0.0, "end": 1.5, "text": "Hello"},
                    {"start": 1.6, "end": 3.0, "text": "World"}
                ]
            }

        Output::

            [
                {
                    "audio_filepath": "sample.wav",
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello"
                },
                {
                    "audio_filepath": "sample.wav",
                    "start": 1.6,
                    "end": 3.0,
                    "text": "World"
                }
            ]
    
    .. admonition:: Example 2 (list of primitives)
        
        .. code-block:: yaml
    
            - _target_: sdp.processors.ListToEntries
              input_manifest_file: ${workspace_dir}/input_manifest.json
              output_manifest_file: ${workspace_dir}/output_manifest.json
              field_with_list: "text_chunks"
              output_field: "text"
                
        Input::
 
            {
                "audio_filepath": "sample.wav",
                "text_chunks": [
                    "Hello",
                    "World"
                ]
            }

        Output::

            [
                {
                    "audio_filepath": "sample.wav",
                    "text": "Hello"
                },
                {
                    "audio_filepath": "sample.wav",
                    "text": "World"
                }
            ]

    """

    def __init__(self, 
        field_with_list: str,
        output_field: str = None,
        **kwargs):
        super().__init__(**kwargs)
        self.field_with_list = field_with_list
        self.output_field = output_field

    def process_dataset_entry(self, data_entry):
        _entries = []

        # Check that the target field is actually a list
        if not isinstance(data_entry[self.field_with_list], list):
            raise TypeError(f'Values of {self.field_with_list} field should be list type only: {data_entry}')
        
        # Remove the list field from the entry and get the list of items
        items_list = data_entry.pop(self.field_with_list)

        # If items are not dicts, output_field must be specified to store the item
        if not isinstance(items_list[0], dict) and not self.output_field:
            raise ValueError(f'Type of items in items list `{self.field_with_list}` is not dict ({type(items_list[0])}). In this case `output_field` should be provided.')
        
        # Expand the list into multiple entries
        for item in items_list:
            _entry = data_entry.copy()

            # If item is a dict, merge its keys; otherwise, store it in `output_field`
            if isinstance(item, dict):
                _entry.update(item)
            else: 
                _entry[self.output_field] = item

            _entry = DataEntry(_entry)
            _entries.append(_entry)

        return _entries


class LambdaExpression(BaseParallelProcessor):
    """
    A dataset processor that evaluates a Python expression on each data entry and either stores
    the result in a new field or uses it as a filtering condition.

    This processor is useful for dynamic field computation or conditional filtering of entries based
    on configurable expressions. It leverages ``evaluate_expression``, which safely evaluates expressions
    using the abstract syntax tree (AST).

    Filtering behavior:
        If ``filter=True``, the expression is evaluated for each entry. Only entries for which the expression evaluates to ``True`` are kept; all others are filtered out (removed from the output).
        If ``filter=False``, the result of the expression is stored in the field specified by ``new_field`` for each entry (no filtering occurs).

    Examples::

        # Example 1: Filtering entries where the duration is greater than 5.0 seconds
        LambdaExpression(
            new_field="keep",  # This field is ignored when filter=True
            expression="entry['duration'] > 5.0",
            lambda_param_name="entry",
            filter=True
        )
        # Only entries with duration > 5.0 will be kept in the output manifest.

        # Example 2: Adding a new field with the number of words in the text
        LambdaExpression(
            new_field="num_words",
            expression="len(entry['text'].split())",
            lambda_param_name="entry",
            filter=False
        )
        # Each entry will have a new field 'num_words' with the word count of the 'text' field.

    Supported operations:

        The expression supports a safe subset of Python operations, including:

        - Arithmetic: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
        - Comparisons: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``is``, ``is not``
        - Logical: ``and``, ``or``, ``not``
        - Bitwise: ``|``, ``&``, ``^``, ``~``, ``<<``, ``>>``
        - Indexing and slicing: ``entry['key']``, ``entry[0]``, ``entry[1:3]``
        - Conditional (ternary) expressions: ``a if cond else b``
        - List and dict literals: ``[a, b]``, ``{k: v}``
        - Attribute access: ``entry.attr``
        - Function calls (limited): ``max``, ``min``, ``len``, ``sum``, ``abs``, ``sorted``

        For the full list, see the ``OPERATORS`` and ``SAFE_FUNCTIONS`` in :mod:`sdp.utils.apply_operators`.
        See also: https://docs.python.org/3/library/operator.html

    Args:
        new_field (str): The name of the field to store the result of the expression (ignored if filter=True).
        expression (str): A Python expression to evaluate. It can reference fields of the data entry
            using the name specified by ``lambda_param_name`` (default: 'entry').
        lambda_param_name (str, optional): The name to refer to the current data entry in the expression.
            Default is "entry".
        filter (bool, optional): If True, the expression result is treated as a condition.
            The entry is kept only if the result is ``True``. Default is ``False``.
        **kwargs: Additional keyword arguments passed to the ``BaseParallelProcessor`` class.

    Returns:
        str: A line-delimited JSON manifest, where each line is a processed entry.
        The result may contain fewer entries than the input if ``filter=True``.
    """
    def __init__(
        self,
        new_field: str,
        expression: str,
        lambda_param_name: str = "entry",
        filter: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.new_field = new_field
        self.expression = expression
        self.lambda_param_name = lambda_param_name
        self.filter = filter

    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """
        Process a single data entry by evaluating the expression.

        If `filter` is True, the entry is only retained if the expression evaluates to True.
        Otherwise, the result is stored in `new_field`.
        """
        value = evaluate_expression(self.expression,  data_entry, self.lambda_param_name)
        if self.filter:
            if value is not True:
                return []
        data_entry[self.new_field] = value   
        return [DataEntry(data=data_entry)]

    def finalize(self, metrics):
        super().finalize(metrics)


class EstimateBandwidth(BaseParallelProcessor):
    """
    Adds estimated bandwidth to each utterance in the input manifest file.

    Args:
        audio_dir (str): Root directory where audio files are stored.
        input_audio_key (str): Manifest key with relative audio paths.
        output_bandwidth_key (str): Manifest key to store estimated bandwidth in.
        max_seconds (float): The maximum length of audio to use for bandwidth estimation.
            By default, uses the first 30 seconds.
        sample_rate (int): Sample rate to resample audio to before doing bandwidth estimation.
            Defaults to 44100, upsampling the input audio as needed.
        n_fft (int): Number of FFT bins to use for bandwidth estimation. Defaults to 512.
        hop_length (int): Audio frame hop length to use for bandwidth estimation.
            Defaults to 441, corresponding to 0.01 seconds for 44100 sample rate.
        top_db (float): top_db treshhold to use for bandwidth estimation.
        frequency_threshold (float): Bandwidth estimation finds the highest frequency with mean power spectrum that is
            within 'frequency_threshold' dB of its peak power. Defaults to -50 dB.

    Returns:
        This processor estimates the bandwidth of the audio file in the`input_audio_key` field and saves the estimate
            in the output_bandwidth_key` field.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.EstimateBandwidth
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_bandwidth.json
              audio_dir: ${workspace_dir}/audio_22khz
              max_workers: 8
    """

    def __init__(
        self,
        audio_dir: str,
        input_audio_key: str = "audio_filepath",
        output_bandwidth_key: str = "bandwidth",
        max_seconds: float = 30.0,
        sample_rate: int = 44100,
        n_fft: int = 512,
        hop_length: int = 441,
        top_db: float = 100.0,
        frequency_threshold: float = -50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_directory = Path(audio_dir)
        self.input_audio_key = input_audio_key
        self.output_bandwidth_key = output_bandwidth_key
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db
        self.frequency_threshold = frequency_threshold

    def _estimate_bandwidth(self, audio, sample_rate):
        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length, window="blackmanharris")
        power_spec = np.abs(spec) ** 2
        power_spec = np.mean(power_spec, axis=1)
        power_spec = librosa.power_to_db(power_spec, ref=self.n_fft, top_db=self.top_db)

        bandwidth = 0
        peak = np.max(power_spec)
        freq_width = sample_rate / self.n_fft
        for idx in range(len(power_spec) - 1, -1, -1):
            if power_spec[idx] - peak > self.frequency_threshold:
                bandwidth = idx * freq_width
                break

        return bandwidth

    def process_dataset_entry(self, data_entry):
        audio_filename = data_entry[self.input_audio_key]
        audio_file = self.audio_directory / audio_filename
        audio, sr = librosa.load(path=audio_file, sr=self.sample_rate, duration=self.max_seconds)
        bandwidth = self._estimate_bandwidth(audio=audio, sample_rate=sr)
        data_entry[self.output_bandwidth_key] = int(bandwidth)
        return [DataEntry(data=data_entry)]


class TrimSilence(BaseParallelProcessor):
    """
    Trims leading and trailing silence from audio files using DeepFilterNet's energy-based method.
    
    This processor implements the same silence detection algorithm used by DeepFilterNet:
    - Computes windowed energy in dB
    - Finds first frame above -120dB (with padding)
    - Finds last frame above -100dB (with padding)
    - Trims audio to that range and updates the file
    
    Args:
        audio_dir (str): Root directory where audio files are stored.
        input_audio_key (str): Manifest key with relative audio paths. Defaults to "audio_filepath".
        start_threshold_db (float): Energy threshold for detecting start of speech in dB. Defaults to -120.
        end_threshold_db (float): Energy threshold for detecting end of speech in dB. Defaults to -100.
        start_padding_frames (int): Number of frames to pad before detected start. Defaults to 15.
        end_padding_frames (int): Number of frames to pad after detected end. Defaults to 10.
        min_frames_for_start (int): Minimum frame index before considering start. Defaults to 14.
        min_frames_for_end (int): Minimum frame index from end before considering end. Defaults to 10.
        update_duration (bool): Whether to update duration in manifest after trimming. Defaults to True.
    
    Returns:
        Audio files are trimmed in-place and duration is updated in manifest if update_duration is True.
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.TrimSilence
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_trimmed.json
              audio_dir: ${workspace_dir}/audio_44khz
              max_workers: 4
    """
    
    def __init__(
        self,
        audio_dir: str,
        input_audio_key: str = "audio_filepath",
        start_threshold_db: float = -120.0,
        end_threshold_db: float = -100.0,
        start_padding_frames: int = 15,
        end_padding_frames: int = 10,
        min_frames_for_start: int = 14,
        min_frames_for_end: int = 10,
        update_duration: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_directory = Path(audio_dir)
        self.input_audio_key = input_audio_key
        self.start_threshold_db = start_threshold_db
        self.end_threshold_db = end_threshold_db
        self.start_padding_frames = start_padding_frames
        self.end_padding_frames = end_padding_frames
        self.min_frames_for_start = min_frames_for_start
        self.min_frames_for_end = min_frames_for_end
        self.update_duration = update_duration
    
    def _windowed_energy(self, audio: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
        """Compute windowed energy in dB, matching DeepFilterNet's implementation."""
        # Normalize audio
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Pad for windowing
        pad_size = window_size // 2
        padded = np.pad(audio, (pad_size, pad_size), mode='constant')
        
        # Compute windowed frames
        n_frames = (len(padded) - window_size) // hop_size + 1
        frames = np.array([padded[i*hop_size:i*hop_size+window_size] 
                          for i in range(n_frames)])
        
        # Compute energy: power -> log10 -> dB
        energy = frames ** 2
        energy = energy + 1e-10  # Avoid log(0)
        energy = np.log10(energy)
        energy = np.mean(energy, axis=1) * 20  # Convert to dB
        
        return energy
    
    def _trim_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, bool]:
        """Trim silence from audio using DeepFilterNet's method."""
        window_size = sample_rate // 10  # 100ms windows
        hop_size = sample_rate // 20     # 50ms hop
        
        # Compute windowed energy
        energy = self._windowed_energy(audio, window_size, hop_size)
        
        if len(energy) == 0:
            return audio, False
        
        # Find start: first frame above start_threshold_db
        start_frame = 0
        for i in range(len(energy)):
            if energy[i] > self.start_threshold_db and i > self.min_frames_for_start:
                start_frame = max(0, i - self.start_padding_frames)
                break
        
        # Find end: last frame above end_threshold_db
        end_frame = len(energy)
        for i in range(1, len(energy) + 1):
            if energy[-i] > self.end_threshold_db and i > self.min_frames_for_end:
                end_frame = min(len(energy), len(energy) - i + self.end_padding_frames)
                break
        
        # Check if trimming is needed
        if start_frame >= end_frame or end_frame - start_frame >= len(energy):
            return np.array([]), True  # Everything would be trimmed
        
        # Convert frame indices to sample indices
        start_sample = start_frame * hop_size
        end_sample = end_frame * hop_size
        
        if end_frame < len(energy) - 10:  # Significant trimming occurred
            trimmed_audio = audio[start_sample:end_sample]
            return trimmed_audio, True
        
        return audio, False
    
    def process_dataset_entry(self, data_entry):
        audio_filename = data_entry[self.input_audio_key]
        audio_file = self.audio_directory / audio_filename
        
        if not audio_file.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            return [DataEntry(data=data_entry)]
        
        try:
            # Load audio at native sample rate
            audio, sr = librosa.load(path=str(audio_file), sr=None)
            
            # Trim silence
            trimmed_audio, was_trimmed = self._trim_audio(audio, sr)
            
            if len(trimmed_audio) == 0:
                logger.warning(f"Audio {audio_file} was completely trimmed, skipping")
                return []  # Drop this entry
            
            if was_trimmed:
                # Save trimmed audio back to file
                soundfile.write(file=str(audio_file), data=trimmed_audio, samplerate=int(sr))
                
                # Update duration in manifest
                if self.update_duration:
                    new_duration = len(trimmed_audio) / sr
                    data_entry["duration"] = round(new_duration, 3)
            
            return [DataEntry(data=data_entry)]
            
        except Exception as e:
            logger.warning(f"Failed to trim silence from {audio_file}: {e}")
            return [DataEntry(data=data_entry)]


class CharacterHistogramLangValidator(BaseParallelProcessor):
    """
    A processor that filters text based on character histogram similarity to trusted data in the target language.

    This processor computes the ratio of characters in a given text that are found in a reference character histogram
    for a specific language. If this ratio is below a certain threshold, the text is likely mislabeled or noisy.

    Histograms are sourced from the NLLB paper (https://arxiv.org/pdf/2207.04672), see page 30 for methodology. This
    technique is a lightweight language ID filter, designed to catch mismatches between text content and claimed language.

    Reference implementation: https://github.com/facebookresearch/fairseq/blob/main/examples/m2m_100/process_data/clean_histogram.py

    Args:
        text_field (str): Key in the data entry containing the text to evaluate.
        lang_field (str, optional): Key in the data entry that identifies the language. Required if `lang` is not provided.
        lang (str, optional): Language code to use for all entries (overrides `lang_field`). Required if `lang_field` is not provided.
        threshold (float): Threshold ratio to determine if text matches the histogram. Used only externally (not enforced in this processor).
        cache_dir (str, optional): Directory where histograms are downloaded and cached.
        threshold_char (str): Character used to truncate the histogram file (default is ']').
        output_score_field (str): Key name under which the computed character match ratio will be stored.
        **kwargs: Additional keyword arguments passed to `BaseParallelProcessor`.

    Raises:
        ValueError: If both `lang` and `lang_field` are provided, or if neither is provided.
                    Also raised if histogram for specified language is missing.

    Returns:
        A manifest where each entry includes the additional field `output_score_field` with the character match ratio.
            Example::

                {
                    "text": "hello world",
                    "lang": "en",
                    "hist_token_ratio": 0.95
                }
    """

    HISTOGRAMS_URL = 'https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz'

    def __init__(self,
                 text_field: str,
                 lang_field: str = None,
                 lang: str = None,
                 threshold: float = 0.8,
                 cache_dir: str = None,
                 threshold_char: str = "]",
                 output_score_field: str = "hist_token_ratio",
                 **kwargs):
        super().__init__(**kwargs)
        self.text_field = text_field

        # Ensure exactly one of `lang` or `lang_field` is provided
        if lang_field is None and lang is None: 
            raise ValueError("One of the arguments `lang` or `lang_field` must be provided.")
        if lang_field is not None and lang is not None: 
            raise ValueError(
                f"Both `lang` ({lang}) and `lang_field` ({lang_field}) are provided, which makes the source of language ambiguous. Please provide only one of them."
            )

        self.lang_field = lang_field
        self.lang = lang
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.threshold_char = threshold_char
        self.output_score_field = output_score_field
        self.histograms = dict()

    def _read_hist(self, lang: str):
        """
        Read and parse the histogram file for a given language, stopping at the threshold character.
        """
        hist_file = os.path.join(self.cache_dir, lang)
        chars = []
        with open(hist_file) as hist:
            for line in hist:
                char = line[0] 
                chars.append(char)
                if char == self.threshold_char:
                    break
        self.histograms[lang] = set(chars)

    def _download_histograms(self):
        """
        Download and extract histogram files into the cache directory.
        """
        logger.info('Downloading histograms collection..')
        response = requests.get(self.HISTOGRAMS_URL)
        if response.status_code != 200:
            raise requests.exceptions.RequestException(
                f"Failed to download model file. Status code: {response.status_code}"
            )

        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()
        
        os.makedirs(self.cache_dir, exist_ok=True)

        histograms_tarfile = wget.download(self.HISTOGRAMS_URL, out=self.cache_dir)
        with tarfile.open(histograms_tarfile, "r:gz") as tar:
            tar.extractall(path=self.cache_dir)

        # Flatten subdirectories into the main cache_dir
        histograms_filepaths = glob(f'{self.cache_dir}/checkpoint/edunov/cc60_multilingual/clean_hists/*')
        for histogram_filepath in histograms_filepaths:
            shutil.move(histogram_filepath, os.path.join(self.cache_dir, os.path.basename(histogram_filepath)))

        os.remove(histograms_tarfile)
        shutil.rmtree(f'{self.cache_dir}/checkpoint/edunov/cc60_multilingual/clean_hists/')
        logger.info(f'Histograms have been downloaded to {self.cache_dir}.')

    def prepare(self):
        """
        Ensure histograms are available and read them into memory.
        """
        if (self.cache_dir is None or 
            not os.path.exists(self.cache_dir) or 
            not os.path.isdir(self.cache_dir) or 
            len(os.listdir(self.cache_dir)) == 0):
            
            self._download_histograms()

        logger.info('Reading histograms...')
        available_langs = os.listdir(self.cache_dir)
        if self.lang is not None:
            if self.lang in available_langs:
                self._read_hist(self.lang)
            else:
                raise ValueError(f"Invalid value for `lang`: {self.lang}. Please provide one of the following: {available_langs}")
            logger.info(f'Histogram for `{self.lang}` has been read.')
        else:
            for lang in tqdm(available_langs):
                self._read_hist(lang)
            logger.info(f'Histograms have been read.')

    def process_dataset_entry(self, data_entry):
        """
        Compute and attach the character histogram match ratio for a given text entry.

        Args:
            data_entry (dict): A dictionary containing at least `text_field` and either `lang_field` or a preset `lang`.

        Returns:
            List[DataEntry]: A list with one updated `DataEntry` including the character match ratio field.
        """
        # Determine language for this entry
        lang = self.lang if self.lang is not None else data_entry[self.lang_field]
        if lang not in self.histograms:
            raise ValueError(f'lang `{lang}` is not supported.')

        # Compute how many characters match the histogram
        text = data_entry[self.text_field].strip()
        cnt = len([c for c in text if c in self.histograms[lang]])
        token_ratio = cnt / len(text) if len(text) > 0 else 0.0

        # Store the ratio in the data entry
        data_entry[self.output_score_field] = token_ratio
        return [DataEntry(data=data_entry)]


class FilterAndSpeakerSplitManifest(BaseProcessor):
    """
    Filters entries shorter than ``min_duration`` seconds and assigns a speaker-grouped
    train / test / val split into the ``set_key`` field.

    Splitting is done at the **speaker** level to avoid leakage: each speaker is
    assigned to exactly one split. Speakers are shuffled with a fixed seed and
    greedily assigned to match the desired utterance-level ratios as closely as
    possible.

    Args:
        min_duration (float): Minimum duration in seconds to keep an utterance.
        train_ratio (float): Fraction of utterances to allocate to the train split.
        test_ratio (float): Fraction of utterances to allocate to the test split.
        val_ratio (float): Fraction of utterances to allocate to the validation split.
        speaker_key (str): Manifest key containing speaker ID.
        set_key (str): Manifest key to write the split label into.
        seed (int): Random seed for speaker shuffling (for reproducibility).

    Returns:
        A single manifest written to ``output_manifest_file`` with:
          - all entries having duration >= ``min_duration``
          - a ``set_key`` field set to one of ``{\"train\",\"test\",\"val\"}``.
    """

    def __init__(
        self,
        min_duration: float = 3.0,
        train_ratio: float = 0.75,
        test_ratio: float = 0.15,
        val_ratio: float = 0.10,
        speaker_key: str = "speaker",
        set_key: str = "set",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_duration = float(min_duration)
        self.ratios = {
            "train": float(train_ratio),
            "test": float(test_ratio),
            "val": float(val_ratio),
        }
        total = sum(self.ratios.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        self.speaker_key = speaker_key
        self.set_key = set_key
        self.seed = int(seed)

    def process(self):
        manifest = load_manifest(self.input_manifest_file)
        if not manifest:
            logger.warning("Input manifest is empty; writing empty output.")
            save_manifest([], self.output_manifest_file)
            return

        # Filter by duration first
        filtered = []
        for entry in manifest:
            if "duration" not in entry:
                raise KeyError("Expected 'duration' key in manifest entry but none found.")
            if entry["duration"] is None:
                continue
            try:
                dur = float(entry["duration"])
            except Exception as e:
                raise ValueError(f"Non-numeric duration value {entry['duration']}") from e
            if dur >= self.min_duration:
                if self.speaker_key not in entry:
                    raise KeyError(
                        f"Expected speaker key '{self.speaker_key}' in manifest entry but none found."
                    )
                filtered.append(entry)

        if not filtered:
            logger.warning(
                "No entries remaining after duration filtering (min_duration=%.3f).",
                self.min_duration,
            )
            save_manifest([], self.output_manifest_file)
            return

        # Group by speaker
        by_speaker: Dict[str, List[dict]] = {}
        for e in filtered:
            spk = str(e[self.speaker_key])
            by_speaker.setdefault(spk, []).append(e)

        speakers = list(by_speaker.keys())
        # Sort by size (largest first) to assign large speakers to large splits first
        speakers.sort(key=lambda spk: len(by_speaker[spk]), reverse=True)
        rng = np.random.RandomState(self.seed)
        # Shuffle speakers of the same size to maintain some randomness
        # Group by size, shuffle within groups, then flatten
        from collections import defaultdict
        by_size = defaultdict(list)
        for spk in speakers:
            by_size[len(by_speaker[spk])].append(spk)
        speakers = []
        for size in sorted(by_size.keys(), reverse=True):
            same_size = by_size[size]
            rng.shuffle(same_size)
            speakers.extend(same_size)

        total_utts = sum(len(v) for v in by_speaker.values())
        targets = {
            split: self.ratios[split] * total_utts
            for split in ("train", "test", "val")
        }
        counts = {split: 0 for split in ("train", "test", "val")}
        speaker_to_split: Dict[str, str] = {}

        for spk in speakers:
            # Greedy assignment using relative deviation to handle skewed sizes
            best_split = None
            best_score = None
            spk_size = len(by_speaker[spk])
            for split in ("train", "test", "val"):
                if self.ratios[split] == 0:
                    continue  # Skip splits with zero ratio
                projected = counts[split] + spk_size
                # Use relative deviation: |projected - target| / (target + 1) to avoid division by zero
                # This ensures splits with larger targets (higher ratios) are prioritized
                target = targets[split]
                if target > 0:
                    score = abs(projected - target) / (target + 1)
                else:
                    score = float('inf')
                if best_score is None or score < best_score:
                    best_score = score
                    best_split = split
            # Fallback: if somehow no split was chosen, assign to train
            if best_split is None:
                best_split = "train"
            speaker_to_split[spk] = best_split
            counts[best_split] += spk_size

        for e in filtered:
            spk = str(e[self.speaker_key])
            split = speaker_to_split[spk]
            e[self.set_key] = split

        logger.info(
            "FilterAndSpeakerSplitManifest: kept %d entries (out of %d), "
            "assigned splits (train=%d, test=%d, val=%d)",
            len(filtered),
            len(manifest),
            counts["train"],
            counts["test"],
            counts["val"],
        )
        save_manifest(filtered, self.output_manifest_file)


class ExportPersonalizationArtifacts(BaseProcessor):
    """
    Export per-split manifests and DeepFilterNet-style HDF5 + JSON sidecars
    from a single split-annotated manifest.

    This processor assumes:
      - Each entry has an audio path in ``audio_key``.
      - Each entry has a speaker ID in ``speaker_key``.
      - Each entry has a split label in ``set_key`` (e.g., \"train\", \"test\", \"val\").

    For each requested split, it produces:
      - ``manifest_<split>.json``: NeMo manifest containing only that split.
      - ``<hdf5_basename>_<split>.hdf5``: HDF5 with group \"/speech\" and one
        dataset per utterance (PCM samples), plus ``n_samples`` attribute.
      - ``<hdf5_basename>_<split>.hdf5.speaker_ids.json``:
            {\"<hdf5_filename>\": {\"<key>\": \"<speaker_id>\", ...}}
      - ``<hdf5_basename>_<split>.hdf5.speaker_index.json``:
            {\"speaker_to_indices\": ..., \"index_to_speaker\": [...], \"total_samples\": N}

    HDF5 keys follow the same convention as in ``REFERENCE/prepare_data.py``:
        rel_path = os.path.relpath(audio_path, base_audio_dir)
        key      = rel_path.replace(\"/\", \"_\")
    """

    def __init__(
        self,
        base_audio_dir: str,
        output_dir: str,
        audio_key: str = "audio_filepath",
        speaker_key: str = "speaker",
        set_key: str = "set",
        splits: Optional[List[str]] = None,
        sample_rate: int = 48000,
        dtype: str = "int16",
        codec: str = "pcm",
        codec_compression: Optional[int] = None,
        copy_encoded_if_compatible: bool = False,
        hdf5_basename: str = "speech",
        max_workers: int = -1,
        resume_if_exists: bool = False,
        preprocess_batch_size: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_audio_dir = os.path.abspath(base_audio_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.audio_key = audio_key
        self.speaker_key = speaker_key
        self.set_key = set_key
        self.splits = splits or ["train", "test", "val"]
        self.sample_rate = int(sample_rate)
        self.dtype = dtype
        self.codec = codec.lower()
        self.codec_compression = codec_compression
        self.copy_encoded_if_compatible = bool(copy_encoded_if_compatible)
        self.hdf5_basename = hdf5_basename
        self.resume_if_exists = bool(resume_if_exists)
        self.preprocess_batch_size = max(1, int(preprocess_batch_size))
        if max_workers == -1:
            max_workers = os.cpu_count() or 1
        self.max_workers = max(1, int(max_workers))

        if self.codec not in ("pcm", "flac", "vorbis"):
            raise NotImplementedError(
                f"Only 'pcm', 'flac', and 'vorbis' codecs are supported in ExportPersonalizationArtifacts, got '{self.codec}'."
            )
        if self.dtype not in ("int16", "float32"):
            raise ValueError("dtype must be 'int16' or 'float32'")
        if self.codec == "vorbis" and self.dtype == "int16":
            logger.warning("Vorbis codec stores encoded bytes; dtype=%s metadata is ignored.", self.dtype)

    def _can_passthrough_encoded_audio(self, path: str) -> bool:
        """Return True when we can copy encoded bytes without decoding/re-encoding."""
        if not self.copy_encoded_if_compatible:
            return False
        if self.codec == "flac" and not path.lower().endswith(".flac"):
            return False
        if self.codec == "vorbis" and not (path.lower().endswith(".ogg") or path.lower().endswith(".vorbis")):
            return False

        try:
            meta = torchaudio.info(path)
            # Keep output behavior compatible with exporter assumptions: mono + target sample rate.
            if int(meta.sample_rate) != self.sample_rate:
                return False
            if int(getattr(meta, "num_channels", 1)) != 1:
                return False
            return True
        except Exception:
            return False

    def _entry_to_key_and_speaker(self, entry: dict) -> Tuple[str, str, str]:
        rel_path = entry[self.audio_key]
        audio_path = os.path.join(self.base_audio_dir, rel_path)
        key = os.path.relpath(audio_path, self.base_audio_dir).replace(os.sep, "_")
        speaker = str(entry[self.speaker_key])
        return audio_path, key, speaker

    def _prepare_entry_for_hdf5(self, entry: dict) -> Tuple[str, np.ndarray, str, int]:
        audio_path, key, speaker = self._entry_to_key_and_speaker(entry)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, n_samples = self._audio_to_array(audio_path)
        return key, audio, speaker, n_samples

    def _iter_prepared_entries(self, entries: List[dict]):
        if self.max_workers == 1:
            for entry in entries:
                yield self._prepare_entry_for_hdf5(entry)
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for prepared in executor.map(self._prepare_entry_for_hdf5, entries):
                yield prepared

    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _iter_manifest_entries(self, manifest_path: str):
        with open(manifest_path, "rt", encoding="utf8") as fin:
            for line in fin:
                if line.strip():
                    yield json.loads(line)

    def _iter_manifest_batches(self, manifest_path: str, batch_size: int):
        batch = []
        for entry in self._iter_manifest_entries(manifest_path):
            batch.append(entry)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _audio_to_array(self, path: str) -> Tuple[np.ndarray, int]:
        if self.codec in ("flac", "vorbis") and self._can_passthrough_encoded_audio(path):
            meta = torchaudio.info(path)
            with open(path, "rb") as fin:
                encoded = np.frombuffer(fin.read(), dtype=np.uint8).copy()
            return encoded, int(meta.num_frames)

        # Load audio; resample to target sample_rate if needed.
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        # (channels, samples) -> (samples,) mono
        if waveform.dim() == 2:
            waveform = waveform[0]
        n_samples = int(waveform.shape[0])

        if self.codec == "pcm":
            audio = waveform.numpy()
            if self.dtype == "int16":
                # Normalize to [-1,1] if not already, then scale.
                max_val = np.max(np.abs(audio)) or 1.0
                audio = audio / max_val
                audio = (audio * 32767.0).astype(np.int16)
            else:
                audio = audio.astype(np.float32)
            return audio, n_samples

        if self.codec == "flac":
            compression = 8 if self.codec_compression is None else int(self.codec_compression)
            with NamedTemporaryFile(suffix=".flac") as tf:
                torchaudio.save(
                    tf.name,
                    waveform.reshape(1, -1),
                    self.sample_rate,
                    format="flac",
                    compression=compression,
                    bits_per_sample=16,
                )
                tf.seek(0)
                encoded = np.frombuffer(tf.read(), dtype=np.uint8).copy()
            return encoded, n_samples

        if self.codec == "vorbis":
            compression = 8 if self.codec_compression is None else int(self.codec_compression)
            with NamedTemporaryFile(suffix=".ogg") as tf:
                torchaudio.save(
                    tf.name,
                    waveform.reshape(1, -1),
                    self.sample_rate,
                    format="vorbis",
                    compression=compression,
                )
                tf.seek(0)
                encoded = np.frombuffer(tf.read(), dtype=np.uint8).copy()
            return encoded, n_samples

        raise NotImplementedError(f"Codec '{self.codec}' not supported.")

    def _write_split_artifacts(self, split: str, manifest_path: str, num_entries: int):
        if num_entries == 0:
            logger.warning("No entries for split '%s'; skipping artifact creation.", split)
            return

        # HDF5 audio
        hdf5_filename = f"{self.hdf5_basename}_{split}.hdf5"
        hdf5_path = os.path.join(self.output_dir, hdf5_filename)
        sidecar_ids_path = hdf5_path + ".speaker_ids.json"
        sidecar_index_path = hdf5_path + ".speaker_index.json"
        speaker_to_indices: Dict[str, array] = {}

        # Keep this stream on disk to avoid storing a potentially huge list of speakers in RAM.
        tmp_fd, index_to_speaker_tmp_path = tempfile.mkstemp(
            prefix=f"{split}_index_to_speaker_", suffix=".tmp"
        )
        os.close(tmp_fd)
        tmp_idx_to_speaker = open(index_to_speaker_tmp_path, "w", encoding="utf8")

        try:
            with open(sidecar_ids_path, "w", encoding="utf8") as ids_out:
                ids_out.write("{")
                ids_out.write(json.dumps(hdf5_filename))
                ids_out.write(":{")
                first_pair = True

                hdf5_mode = "a" if self.resume_if_exists and os.path.exists(hdf5_path) else "w"
                with h5.File(hdf5_path, hdf5_mode) as f:
                    if "speech" in f:
                        grp = f["speech"]
                    else:
                        grp = f.create_group("speech")
                    f.attrs["sr"] = self.sample_rate
                    f.attrs["dtype"] = self.dtype
                    f.attrs["codec"] = self.codec

                    output_index = 0
                    written_count = 0
                    reused_count = 0
                    for batch in self._iter_manifest_batches(manifest_path, batch_size=self.preprocess_batch_size):
                        batch_meta = []
                        entries_to_prepare = []
                        for entry in batch:
                            _, key, speaker = self._entry_to_key_and_speaker(entry)
                            exists_in_hdf5 = key in grp
                            batch_meta.append((key, speaker, exists_in_hdf5))
                            if not exists_in_hdf5:
                                entries_to_prepare.append(entry)

                        prepared_iter = iter(self._iter_prepared_entries(entries_to_prepare))
                        for key, speaker, exists_in_hdf5 in batch_meta:
                            if exists_in_hdf5:
                                reused_count += 1
                            else:
                                prepared_key, audio, prepared_speaker, n_samples = next(prepared_iter)
                                if prepared_key != key:
                                    raise RuntimeError(
                                        f"Prepared key mismatch: expected {key}, got {prepared_key}"
                                    )
                                ds = grp.create_dataset(prepared_key, data=audio, compression=None)
                                ds.attrs["n_samples"] = n_samples
                                speaker = prepared_speaker
                                written_count += 1

                            if not first_pair:
                                ids_out.write(",")
                            first_pair = False
                            ids_out.write(json.dumps(key))
                            ids_out.write(":")
                            ids_out.write(json.dumps(speaker))

                            tmp_idx_to_speaker.write(speaker)
                            tmp_idx_to_speaker.write("\n")
                            speaker_to_indices.setdefault(speaker, array("I")).append(output_index)
                            output_index += 1

                ids_out.write("}}")

            with open(sidecar_index_path, "w", encoding="utf8") as fout:
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
                total_samples = 0
                first_index = True
                with open(index_to_speaker_tmp_path, "r", encoding="utf8") as fin:
                    for line in fin:
                        speaker = line.rstrip("\n")
                        if not first_index:
                            fout.write(",")
                        first_index = False
                        fout.write(json.dumps(speaker))
                        total_samples += 1
                fout.write("],")
                fout.write(f"\"total_samples\":{total_samples}")
                fout.write("}")
        finally:
            tmp_idx_to_speaker.close()
            if os.path.exists(index_to_speaker_tmp_path):
                os.remove(index_to_speaker_tmp_path)

        logger.info(
            "Exported split '%s': %d entries (written=%d, reused=%d, resume=%s) -> %s, %s, %s",
            split,
            num_entries,
            written_count,
            reused_count,
            self.resume_if_exists,
            manifest_path,
            sidecar_ids_path,
            sidecar_index_path,
        )

    def process(self):
        self._ensure_output_dir()
        if not self.input_manifest_file or not os.path.exists(self.input_manifest_file):
            logger.warning("ExportPersonalizationArtifacts input manifest not found: %s", self.input_manifest_file)
            return

        split_manifest_paths = {
            split: os.path.join(self.output_dir, f"manifest_{split}.json") for split in self.splits
        }
        split_counts = {split: 0 for split in self.splits}
        split_handles = {
            split: open(path, "w", encoding="utf8") for split, path in split_manifest_paths.items()
        }
        try:
            for entry in self._iter_manifest_entries(self.input_manifest_file):
                split = entry.get(self.set_key)
                if split not in split_handles:
                    continue
                json.dump(entry, split_handles[split], ensure_ascii=False)
                split_handles[split].write("\n")
                split_counts[split] += 1
        finally:
            for handle in split_handles.values():
                handle.close()

        if sum(split_counts.values()) == 0:
            logger.warning("ExportPersonalizationArtifacts found no recognized split entries; nothing to export.")
            return

        for split in self.splits:
            self._write_split_artifacts(
                split=split,
                manifest_path=split_manifest_paths[split],
                num_entries=split_counts[split],
            )

        # For compatibility with BaseProcessor, set output_manifest_file to the train manifest if it exists
        train_manifest = os.path.join(self.output_dir, "manifest_train.json")
        if os.path.exists(train_manifest):
            # overwrite primary output with the train manifest path if user pointed elsewhere
            if self.output_manifest_file and self.output_manifest_file != train_manifest:
                shutil.copy(train_manifest, self.output_manifest_file)
