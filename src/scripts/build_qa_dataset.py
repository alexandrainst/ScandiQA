"""Script that builds the ScandiQA dataset."""

from typing import List

from scandi_qa import QADatasetBuilder


def main(languages: List[str] = ["da", "sv", "no"]):
    """Build the ScandiQA dataset.

    Args:
        languages (list of str, optional):
            The languages to build the dataset for. Defaults to ["da", "sv", "no"].
    """
    # Build the dataset
    for language in languages:
        builder = QADatasetBuilder(language=language)
        builder.build()


if __name__ == "__main__":
    main()
