"""Caching classes for use in other modules."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


class TranslationCache:
    """A class for caching translations.

    Args:
        cache_name (str):
            The name of the cache. The cache will be stored at
            data/processed/{cache_name}.jsonl.
    """

    def __init__(self, cache_name: str = "translation_cache") -> None:
        self.cache_path = Path("data") / "processed" / f"{cache_name}.jsonl"
        self.cache: dict = defaultdict(dict)

        # If the cache file doesn't exist then create an empty cache
        if not self.cache_path.exists():
            self.cache_path.touch()

        # Otherwise, load the existing cache
        else:
            # Load the records from the cache file
            with Path(self.cache_path).open("r") as f:
                records = [json.loads(record) for record in f]

            # Convert the records to a dataframe
            df = pd.DataFrame.from_records(records)

            # Extract the languages in the cache
            languages = sorted(df.target_lang.unique().tolist())

            # Load the cache for each language
            for language in languages:
                self.cache[language] = {
                    row.context_en: row.context
                    for _, row in df.query("target_lang == @language").iterrows()
                }

    def add_to_cache(self, text: str, translation: str, target_lang: str) -> None:
        """Add an entry to the cache.

        Args:
            text (str):
                The text to translate.
            translation (str):
                The translation of the text.
            target_lang (str):
                The target language of the translation.
        """
        # Save the translation to the in-memory cache
        self.cache[target_lang][text] = translation

        # Append the translation to the cache file
        with self.cache_path.open("a") as f:
            record = dict(context_en=text, context=translation, target_lang=target_lang)
            f.write(json.dumps(record) + "\n")

    def contains(self, text: str, target_lang: str) -> bool:
        """Check if the text is in the cache.

        Args:
            text (str):
                The text to check.
            target_lang (str):
                The target language of the text.

        Returns:
            bool:
                Whether the text is in the cache.
        """
        return text in self.cache[target_lang]

    def get_translation(self, text: str, target_lang: str) -> str:
        """Get the translation for the text.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The target language of the translation.

        Returns:
            str:
                The translation of the text.
        """
        return self.cache[target_lang][text]
