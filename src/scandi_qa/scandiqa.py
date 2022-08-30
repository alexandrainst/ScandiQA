# Copyright 2022 The HuggingFace Datasets Authors and Dan Saattrup Nielsen.
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
"""Python build script for the ScandiQA dataset."""


import json
from pathlib import Path
from typing import List

from datasets import Version
from datasets.builder import BuilderConfig, GeneratorBasedBuilder
from datasets.download import DownloadManager
from datasets.features import Features, Value
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator

_DESCRIPTION = """
ScandiQA is a dataset of questions and answers in the Danish, Norwegian, and Swedish
languages. All samples come from the Natural Questions (NQ) dataset, which is a large
question answering dataset from Google searches. The Scandinavian questions and answers
come from the MKQA dataset, where 10,000 NQ samples were manually translated into,
among others, Danish, Norwegian, and Swedish. However, this did not include a
translated context, hindering the training of extractive question answering models.

We merged the NQ dataset with the MKQA dataset, and extracted contexts as either "long
answers" from the NQ dataset, being the paragraph in which the answer was found, or
otherwise we extract the context by locating the paragraphs which have the largest
cosine similarity to the question, and which contains the desired answer.

Further, many answers in the MKQA dataset were "language normalised": for instance, all
date answers were converted to the format "YYYY-MM-DD", meaning that in most cases
these answers are not appearing in any paragraphs. We solve this by extending the MKQA
answers with plausible "answer candidates", being slight perturbations or translations
of the answer.

With the contexts extracted, we translated these to Danish, Swedish and Norwegian using
the DeepL translation service for Danish and Swedish, and the Google Translation
service for Norwegian. After translation we ensured that the Scandinavian answers do
indeed occur in the translated contexts.

As we are filtering the MKQA samples at both the "merging stage" and the "translation
stage", we are not able to fully convert the 10,000 samples to the Scandinavian
languages, and instead get roughly 8,000 samples per language. These have further been
split into a training, validation and test split, with the former two containing
roughly 750 samples. The splits have been created in such a way that the proportion of
samples without an answer is roughly the same in each split.
"""

_HOMEPAGE = "https://huggingface.co/alexandrainst/scandiqa"
_LICENSE = "CC BY 4.0"
_URLS = {
    "da": [
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/da/train.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/da/val.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/da/test.jsonl",
    ],
    "sv": [
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/sv/train.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/sv/val.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/sv/test.jsonl",
    ],
    "no": [
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/no/train.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/no/val.jsonl",
        "https://huggingface.co/datasets/saattrupdan/scandiqa/resolve/main/data/no/test.jsonl",
    ],
}

# _CITATION = """
# @InProceedings{huggingface:dataset,
# title = {ScandiQA: A Scandinavian Question Answering Dataset},
# author={Dan Saattrup Nielsen},
# year={2022}
# }
# """


class ScandiQA(GeneratorBasedBuilder):
    """Scandinavian question answering dataset."""

    VERSION = Version("1.0.0")

    BUILDER_CONFIGS = [
        BuilderConfig(
            name="da",
            version=VERSION,
            description="The Danish part of the ScandiQA dataset.",
        ),
        BuilderConfig(
            name="sv",
            version=VERSION,
            description="The Swedish part of the ScandiQA dataset.",
        ),
        BuilderConfig(
            name="no",
            version=VERSION,
            description="The Norwegian part of the ScandiQA dataset.",
        ),
    ]

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "example_id": Value("int64"),
                "question": Value("string"),
                "answer": Value("string"),
                "answer_start": Value("int64"),
                "context": Value("string"),
                "answer_en": Value("string"),
                "answer_start_en": Value("int64"),
                "context_en": Value("string"),
                "title_en": Value("string"),
            }
        )
        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
        urls = _URLS[self.config.name]
        downloaded_files = dl_manager.download_and_extract(urls)
        return [
            SplitGenerator(
                name=str(Split.TRAIN),
                gen_kwargs=dict(
                    filepath=downloaded_files[0],
                    split="train",
                ),
            ),
            SplitGenerator(
                name=str(Split.VALIDATION),
                gen_kwargs=dict(
                    filepath=downloaded_files[1],
                    split="val",
                ),
            ),
            SplitGenerator(
                name=str(Split.TEST),
                gen_kwargs=dict(filepath=downloaded_files[2], split="test"),
            ),
        ]

    def _generate_examples(self, filepath: str, split):
        with Path(filepath).open(encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, data
