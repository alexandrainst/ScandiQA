"""Loading and processing of data."""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from .answer_extraction import extract_answer
from .merger import Merger
from .translation import DeepLTranslator, GoogleTranslator, Translator
from .utils import DEEPL_LANGUAGES, MKQA_LANGUAGES


class QADatasetBuilder:
    """Class that builds a QA dataset from the MKQA and NQ datasets.

    Args:
        language (str, optional):
            The two-character language code of the dataset. Defaults to "da".
        cache_dir (str, optional):
            The directory to cache the dataset. Defaults to
            '~/.cache/huggingface/datasets'.

    Attributes:
        language (str):
            The language of the dataset.
        cache_dir (str):
            The directory to cache the dataset.
        merger (Merger):
            The merger used to merge MKQA and NQ.
        translator (DeepLTranslator):
            The translator used to translate the questions.

    Raises:
        ValueError:
            If the language is not supported.
    """

    def __init__(
        self, language: str = "da", cache_dir: str = "~/.cache/huggingface/datasets"
    ):
        # If the language is not supported, raise an error
        if language not in MKQA_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. We only support "
                f"{', '.join(MKQA_LANGUAGES)}"
            )

        self.language = language
        self.cache_dir = cache_dir
        self.merger = Merger(language=self.language, cache_dir=self.cache_dir)

        # Set up translator, depending on the language
        self.translator: Translator
        if language in DEEPL_LANGUAGES:
            self.translator = DeepLTranslator()
        else:
            self.translator = GoogleTranslator()

    def build(self) -> pd.DataFrame:
        """Builds the dataset and pushes it to the Hugging Face Hub.

        Returns:
            Pandas DataFrame:
                The resulting dataset for the given language.
        """
        # Merge the MKQA and NQ datasets
        df = self.merger.merge()

        # Store the merged dataset
        merged_path = Path("data") / "processed" / f"merged_{self.language}.parquet"
        df.to_parquet(merged_path)

        # Translate the English contexts
        df = self.translate_contexts(df)

        # Push to the Hub
        self.push_to_hub(df)

        return df

    def translate_contexts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translates the English contexts of the MKQA dataset.

        Args:
            df (pd.DataFrame):
                The merged dataset from MKQA and NQ.

        Returns:
            pd.DataFrame:
                The merged dataset with the translated contexts and answers.
        """
        # Translate all the English contexts
        with tqdm(df.iterrows(), total=len(df), desc="Translating contexts") as pbar:
            records = [self._translate_context(example) for _, example in pbar]

        # Convert to a Pandas DataFrame and replace MKQA dataset
        translated_df = pd.DataFrame.from_records(records)

        # Remove the rows with missing answers
        translated_df.dropna(subset="answer", inplace=True)

        # Set datatypes of integer columns
        translated_df.answer_start = translated_df.answer_start.astype(int)
        translated_df.answer_start_en = translated_df.answer_start_en.astype(int)

        # Return the translated dataset
        return translated_df

    def push_to_hub(self, df: pd.DataFrame):
        """Pushes the dataset to the Hugging Face Hub.

        Args:
            df (pd.DataFrame):
                The dataset to push to the Hugging Face Hub.
        """
        # Convert the dataframe to a Hugging Face Dataset object
        dataset = Dataset.from_pandas(df)

        # Drop the index column
        dataset = dataset.remove_columns("__index_level_0__")

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(f"scandiqa-{self.language}")
        return self

    def _translate_context(self, example: pd.Series) -> dict:
        """Translate the English context to the target language.

        Args:
            example (pd.Series):
                The MKQA example with an English context that needs to be translated.

        Returns:
            dict:
                The example with the translated context as the 'context' attribute and
                answer as the 'answer' and 'answer_start' attributes. The answer is
                None if the answer, or variants thereof, does not appear in the
                translated context.
        """
        # Translate the English context
        example["context"] = self.translator(
            example.context_en, target_lang=self.language
        )

        # If the example does not have an MKQA answer then we simply use the translated
        # context as the new context and set the starting index to be -1
        if example.answer == "":
            example["answer_start"] = -1
            example["answer_start_en"] = -1

        # Otherwise, attempt to extract the answer from the translated context
        else:
            answer_dict = extract_answer(
                answer=example.answer, context=example.context, language=self.language
            )

            # If no answer could be extracted then set the answer and answer_start to
            # None
            if answer_dict is None:
                example["answer"] = None
                example["answer_start"] = None

            # Otherwise, we set the answer candidate appearing in the translated
            # context as the answer, and the translated context as the context
            else:
                example["answer"] = answer_dict["answer"]
                example["answer_start"] = answer_dict["answer_start"]

                # If the English answer did not appear in the English context, then
                # check if the answer appears in the English context instead, and use
                # that as the English answer and starting index
                if example.answer_start_en == -1:
                    answer_en_dict = extract_answer(
                        answer=example.answer,
                        context=example.context_en,
                        language=self.language,
                    )
                    if answer_en_dict is not None:
                        example["answer_en"] = answer_en_dict["answer"]
                        example["answer_start_en"] = answer_en_dict["answer_start"]

        # Return the example as a dictionary
        return example.to_dict()
