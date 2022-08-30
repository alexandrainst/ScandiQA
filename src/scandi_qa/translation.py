"""Translation wrappers."""

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import nltk
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from tqdm.auto import tqdm

from .cache import TranslationCache

# Load environment variables
load_dotenv()


class Translator(ABC):
    """Abstract translation class.

    Args:
        api_key (str or None, optional):
            An API key for the given translation API. If None then it is assumed that
            the API key have been specified in the GOOGLE_API_KEY environment variable,
            either directly or within a .env file. Defaults to None.
        progress_bar (bool, optional):
            Whether a progress bar should be shown during translation. Defaults to
            True.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        progress_bar: bool = True,
    ):
        # Load the API key from the environment variable if not specified
        if api_key is None:
            api_key = os.environ[self.env_var]

        # Store the variables
        self.api_key = api_key
        self.progress_bar = progress_bar

        # Initialise the cache
        self.cache = TranslationCache()

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
        if self.cache.contains(text=text, target_lang=target_lang):
            return self.cache.get_translation(text=text, target_lang=target_lang)

        # Query the API for the translation, and try again if the API fails
        while True:
            try:
                response = self._get_response(text=text, target_lang=target_lang)
                break
            except RequestException:
                time.sleep(1)

        # If we sent too many requests or if the service is down, then wait a second
        # and try again
        while response.status_code in [429, 503]:
            time.sleep(1)
            response = self._get_response(text=text, target_lang=target_lang)

        # Split the text up if it is too long
        if response.status_code in {400, 411, 413, 414, 502}:

            # If there are newlines in the text then split by the middle newline
            if "\n" in text:

                # Remove duplicate newlines
                text_modified = re.sub(r"\n+", "\n", text)

                # Locate the indices of all the newlines
                newline_indices = [i for i, c in enumerate(text_modified) if c == "\n"]

                # Choose the median index
                median_index = newline_indices[len(newline_indices) // 2]

                # Split the text by the middle newline
                texts = [
                    text_modified[:median_index],
                    text_modified[median_index + 1 :],
                ]

                # Translate each text
                with tqdm(texts, desc="Translating text chunks", leave=False) as pbar:
                    translations = [
                        self.translate(text=chunk, target_lang=target_lang)
                        for chunk in pbar
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
                        self.translate(
                            text=chunk, target_lang=target_lang, is_sentence=True
                        )
                        for chunk in pbar
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
                        self.translate(
                            text=chunk, target_lang=target_lang, is_sentence=True
                        )
                        for chunk in pbar
                    ]

                # Join the translations together with newlines
                translation = "".join(translations)

        # Otherwise, if the status code is not 200 then raise an error
        elif response.status_code != 200:

            raise RuntimeError(
                f"The translation API returned status code {response.status_code}."
            )

        # Otherwise, we can return the translation
        else:

            # Extract the JSON object from the response
            response_json = response.json()

            # Extract the translation
            translation = self._extract_translation(response_json=response_json)

        # Add the translation to the cache
        self.cache.add_to_cache(
            text=text, translation=translation, target_lang=target_lang
        )

        # Return the translation
        return translation

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

    @property
    @abstractmethod
    def env_var(self) -> str:
        pass

    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @abstractmethod
    def _get_response(self, text: str, target_lang: str) -> requests.Response:
        pass

    @abstractmethod
    def _extract_translation(self, response_json: dict) -> str:
        pass


class DeepLTranslator(Translator):
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

    env_var: str = "DEEPL_API_KEY"
    url: str = "https://api.deepl.com/v2/translate"

    def _get_response(self, text: str, target_lang: str) -> requests.Response:
        """Get the response from the DeepL API.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The language to translate the text into.

        Returns:
            requests.Response:
                The response from the DeepL API.
        """

        # Set up the DeepL API parameters
        params = dict(text=text, auth_key=self.api_key, target_lang=target_lang)

        # Call the DeepL API to get the translations
        response = requests.post(self.url, params=params)  # type: ignore

        # Return the response
        return response

    def _extract_translation(self, response_json: dict) -> str:
        """Extract the translation from the response JSON.

        Args:
            response_json (dict):
                The JSON response from the DeepL API.

        Returns:
            str:
                The translation.
        """

        # Extract the translation
        translation = response_json["translations"][0]["text"]

        # Return the translation
        return translation


class GoogleTranslator(Translator):
    """A wrapper for the Google Translation API.

    API documentation available at
    https://cloud.google.com/translate/docs/reference/rest/v2/translate.

    Args:
        api_key (str or None, optional):
            An API key for the Google Translation API. If None then it is assumed that
            the API key have been specified in the GOOGLE_API_KEY environment variable,
            either directly or within a .env file. Defaults to None.
        progress_bar (bool, optional):
            Whether a progress bar should be shown during translation. Defaults to
            True.
    """

    env_var: str = "GOOGLE_API_KEY"
    url: str = "https://translation.googleapis.com/language/translate/v2"

    def _get_response(self, text: str, target_lang: str) -> requests.Response:
        """Get the response from the DeepL API.

        Args:
            text (str):
                The text to translate.
            target_lang (str):
                The language to translate the text into.

        Returns:
            requests.Response:
                The response from the DeepL API.
        """

        # Set up the Google API parameters
        params = dict(target=target_lang, format="text", key=self.api_key, q=[text])

        # Call the DeepL API to get the translations
        response = requests.post(self.url, params=params)  # type: ignore

        # Return the response
        return response

    def _extract_translation(self, response_json: dict) -> str:
        """Extract the translation from the response JSON.

        Args:
            response_json (dict):
                The JSON response from the DeepL API.

        Returns:
            str:
                The translation.
        """

        # Extract the translation
        translation = response_json["data"]["translations"][0]["translatedText"]

        # Return the translation
        return translation
