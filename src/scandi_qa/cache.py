"""Caching classes for use in other modules."""

import json
from pathlib import Path


class TranslationCache:
    """A class for caching translations.

    Args:
        cache_name (str):
            The name of the cache. The cache will be stored at
            data/processed/{cache_name}.jsonl.
    """

    def __init__(self, cache_name: str = "translation_cache") -> None:
        self.cache_path = Path("data") / "processed" / f"{cache_name}.jsonl"

        # If the cache file exists, load the cache from it. Otherwise, create an empty
        # cache.
        if not self.cache_path.exists():
            self.cache_path.touch()
            self.cache = dict()
        else:
            with Path(self.cache_path).open("r") as f:
                self.cache = {
                    record["context_en"]: record["context"]
                    for record in [json.loads(record) for record in f]
                }

    def add_to_cache(self, text: str, translation: str) -> None:
        """Add an entry to the cache.

        Args:
            text (str):
                The text to translate.
            translation (str):
                The translation of the text.
        """
        # Save the translation to the in-memory cache
        self.cache[text] = translation

        # Append the translation to the cache file
        with self.cache_path.open("a") as f:
            record = dict(context_en=text, context=translation)
            f.write(json.dumps(record) + "\n")

    def __contains__(self, text: str) -> bool:
        """Check if the text is in the cache.

        Args:
            text (str):
                The text to check.

        Returns:
            bool:
                Whether the text is in the cache.
        """
        return text in self.cache

    def __getitem__(self, text: str) -> str:
        """Get the translation for the text.

        Args:
            text (str):
                The text to translate.

        Returns:
            str:
                The translation of the text.
        """
        return self.cache[text]
