"""Script that builds the ScandiQA dataset."""

import sys
from typing import List

from scandi_qa import QADatasetBuilder


def main(languages: List[str] = ["da", "sv", "no"]):
    """Build the ScandiQA dataset.

    Args:
        languages (list of str, optional):
            The languages to build the dataset for. Defaults to ["da", "sv", "no"].
    """
    # Build the dataset
    builder = QADatasetBuilder(languages=languages)
    builder.build()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        languages = sys.argv[1:]
    else:
        languages = ["da", "sv", "no"]

    main(languages)
