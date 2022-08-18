"""DeepL translation wrapper."""

import json
import os
import re
from pathlib import Path
from typing import Optional

import nltk
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()


class DeepLTranslator:
    """A wrapper for the DeepL translation API.

    API documentation available at https://www.deepl.com/docs-api/translating-text/.

    Args:
        api_key (str or None, optional):
            An API key for the DeepL Translation API. If None then it is assumed that
            the API key have been specified in the DEEPL_API_KEY environment variable,
            either directly or within a .env file. Defaults to None.
        progress_bar (bool, optional):
            Whether a progress bar should be shown during translation. Defaults to
            True.
    """

    base_url: str = "https://api.deepl.com/v2/translate"

    def __init__(self, api_key: Optional[str] = None, progress_bar: bool = True):

        # Load the API key from the environment variable if not specified
        if api_key is None:
            api_key = os.environ["DEEPL_API_KEY"]

        # Store the variables
        self.api_key = api_key
        self.progress_bar = progress_bar

        # Load in the cache. If it does not exist then create it.
        self.cache_path = Path("data") / "processed" / "translation_cache.jsonl"
        if not self.cache_path.exists():
            self.cache_path.touch()
            self.cache = dict()
        else:
            with Path(self.cache_path).open("r") as f:
                self.cache = {
                    record["context_en"]: record["context"]
                    for record in [json.loads(record) for record in f]
                }

    def translate(self, text: str, target_lang: str, is_sentence: bool = False) -> str:
        """Translate text into the specified language.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The language to translate the text into.
            is_sentence (bool, optional):
                Whether the text is a sentence. Defaults to False.

        Returns:
            str:
                The translated text.
        """
        # If the text has previously been translated then use the cached translation
        if text in self.cache:
            return self.cache[text]

        # Set up the DeepL API parameters
        params = dict(text=text, auth_key=self.api_key, target_lang=target_lang)

        # Call the DeepL API to get the translations
        response = requests.post(self.base_url, params=params)  # type: ignore

        # Split the text up if it is too long (HTTP error code 414)
        if response.status_code == 414:

            # If there are newlines in the text then split by the middle newline
            if "\n" in text:

                # Remove duplicate newlines
                text = re.sub(r"\n+", "\n", text)

                # Locate the indices of all the newlines
                newline_indices = [i for i, c in enumerate(text) if c == "\n"]

                # Choose the median index
                median_index = newline_indices[len(newline_indices) // 2]

                # Split the text by the middle newline
                texts = [text[:median_index], text[median_index + 1 :]]

                # Translate each text
                with tqdm(texts, desc="Translating text chunks", leave=False) as pbar:
                    translations = [
                        self(text=text, target_lang=target_lang) for text in pbar
                    ]

                # Join the translations together with newlines
                translation = "\n".join(translations)

            # Otherwise, we use the nltk library to split the text into sentences
            elif not is_sentence:

                # Split the text into sentences
                try:
                    texts = nltk.sent_tokenize(text)
                except LookupError:
                    nltk.download("punkt", quiet=True)
                    texts = nltk.sent_tokenize(text)

                # Translate each text
                with tqdm(texts, desc="Translating text chunks", leave=False) as pbar:
                    translations = [
                        self(text=text, target_lang=target_lang, is_sentence=True)
                        for text in pbar
                    ]

                # Join the translations together with newlines
                translation = " ".join(translations)

            # Otherwise, we simply split the text into chunks of 500 characters
            else:

                # Split the text into chunks of 500 characters
                texts = [text[i : i + 500] for i in range(0, len(text) - 500, 500)]

                # Translate each text
                with tqdm(texts, desc="Translating text chunks", leave=False) as pbar:
                    translations = [
                        self(text=text, target_lang=target_lang, is_sentence=True)
                        for text in pbar
                    ]

                # Join the translations together with newlines
                translation = "".join(translations)

        # Otherwise, if the status code is not 200 then raise an error
        elif response.status_code != 200:

            raise RuntimeError(
                f"DeepL API returned status code {response.status_code}."
            )

        # Otheriwse, we can return the translation
        else:

            # Extract the JSON object from the response
            response_json = response.json()

            # Extract the translation
            translation = response_json["translations"][0]["text"]

        # Add the translation to the cache
        self.save_to_cache(text=text, translation=translation)

        # Return the translation
        return translation

    def save_to_cache(self, text: str, translation: str) -> None:
        """Save the translation to the cache.

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

    def __call__(self, text: str, target_lang: str, is_sentence: bool = False) -> str:
        """Translate text into the specified language.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The language to translate the text into.
            is_sentence (bool, optional):
                Whether the text is a sentence. Defaults to False.

        Returns:
            str:
                The translated text.
        """
        return self.translate(
            text=text, target_lang=target_lang, is_sentence=is_sentence
        )
