"""Build the ScandiQA dataset."""

from src.scandi_qa import QADatasetBuilder


def main():
    """Main function."""
    for language in ["da"]:
        builder = QADatasetBuilder(language=language)
        builder.build()


if __name__ == "__main__":
    main()
