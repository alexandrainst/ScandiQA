"""DeepL translation wrapper."""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

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

    base_url: str = "https://api-free.deepl.com/v2/translate"

    def __init__(self, api_key: Optional[str] = None, progress_bar: bool = True):

        # Load the API key from the environment variable if not specified
        if api_key is None:
            api_key = os.environ["DEEPL_API_KEY"]

        # Store the variables
        self.api_key = api_key
        self.progress_bar = progress_bar

    def __call__(self, text: str, target_lang: str) -> str:
        """Translate text into the specified language.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The language to translate the text into.

        Returns:
            str:
                The translated text.
        """
        # Set up the DeepL API parameters
        params = dict(
            text=[text],
            auth_key=self.api_key,
            target_lang=target_lang,
            split_sentences=0,
        )

        # Call the DeepL API to get the translations
        response = requests.get(self.base_url, params=params)  # type: ignore

        # Extract the translation
        translated = response.json()["translations"][0]["text"]

        # Return the translation
        return translated
