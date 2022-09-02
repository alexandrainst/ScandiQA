"""Loading and processing of data."""

import itertools as it
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .answer_extraction import extract_answer
from .merger import Merger
from .translation import DeepLTranslator, GoogleTranslator, Translator
from .utils import DEEPL_LANGUAGES, MKQA_LANGUAGES


class QADatasetBuilder:
    """Class that builds a QA dataset from the MKQA and NQ datasets.

    Args:
        languages (list of str, optional):
            The two-character language codes of the dataset. Defaults to
            ["da", "sv", "no"].
        val_size (int, optional):
            The size of the validation set. Defaults to 500.
        test_size (int, optional):
            The size of the test set. Defaults to 500.
        cache_dir (str, optional):
            The directory to cache the dataset. Defaults to
            '~/.cache/huggingface/datasets'.

    Attributes:
        languages (list of str):
            The languages of the dataset.
        cache_dir (str):
            The directory to cache the dataset.
        mergers (Dict[str, Merger]):
            The mergers used to merge MKQA and NQ.
        translators (Dict[str, Translator]):
            The translators used to translate the questions.

    Raises:
        ValueError:
            If the language is not supported.
    """

    def __init__(
        self,
        languages: List[str] = ["da", "sv", "no"],
        val_size: int = 500,
        test_size: int = 500,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        # If one of the languages is not supported, raise an error
        for language in languages:
            if language not in MKQA_LANGUAGES:
                raise ValueError(
                    f"Language '{language}' not supported. We only support "
                    f"{', '.join(MKQA_LANGUAGES)}"
                )

        self.languages = languages
        self.val_size = val_size
        self.test_size = test_size
        self.cache_dir = cache_dir

        # Set up translator, depending on the language
        deepl_translator = DeepLTranslator()
        google_translator = GoogleTranslator()
        self.translators: Dict[str, Translator] = dict()
        for language in self.languages:
            if language in DEEPL_LANGUAGES:
                self.translators[language] = deepl_translator
            else:
                self.translators[language] = google_translator

        # Set up the mergers
        self.mergers = {
            language: Merger(
                translator=self.translators[language],
                language=language,
                cache_dir=self.cache_dir,
            )
            for language in self.languages
        }

    def build(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Builds the dataset and pushes it to the Hugging Face Hub.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]:
                The resulting datasets for each language and each split. The outer
                dictionary has the languages as keys and the inner dictionary has
                the splits ("train", "val" and "test") as keys.
        """
        # Merge the dataset for each language
        merged_dfs = {
            language: self.build_single(language=language)
            for language in self.languages
        }

        # Split into train, validation and test sets for each language. This results in
        # a dictionary with a key for each language, the values of which are dicts with
        # "train", "val" and "test" keys.
        dataset_dicts = self.split_dataset(merged_dfs)

        # Store all the splits as separate JSONL files
        data_dir = Path("src") / "dataset_repo" / "data"
        for language, dataset_dict in dataset_dicts.items():
            for split, dataset in dataset_dict.items():
                path = data_dir / language / f"{split}.jsonl"
                dataset.to_json(path, orient="records", lines=True)

        # Return the dataset
        return dataset_dicts

    def build_single(self, language: str) -> pd.DataFrame:
        """Builds the dataset for a single language.

        Args:
            language (str):
                The language to build the dataset for.

        Returns:
            Pandas DataFrame:
                The resulting dataset for the given language.
        """
        # Set up the path to the merged dataset
        merged_path = Path("data") / "processed" / f"merged_{language}.parquet"

        # Merge the MKQA and NQ datasets if they haven't been merged yet
        if not merged_path.exists():
            df = self.mergers[language].merge()
            df.to_parquet(merged_path)

        # Otherwise load the merged dataset
        else:
            df = pd.read_parquet(merged_path)

        # Translate the English contexts
        df = self.translate_contexts(df, language=language)

        # Return the dataset
        return df

    def translate_contexts(self, df: pd.DataFrame, language: str) -> pd.DataFrame:
        """Translates the English contexts of the MKQA dataset.

        Args:
            df (pd.DataFrame):
                The merged dataset from MKQA and NQ.
            language (str):
                The language to translate the contexts to.

        Returns:
            pd.DataFrame:
                The merged dataset with the translated contexts and answers.
        """
        # Translate all the English contexts
        with tqdm(df.iterrows(), total=len(df), desc="Translating contexts") as pbar:
            records = [
                self._translate_context(example, language=language)
                for _, example in pbar
            ]

        # Convert to a Pandas DataFrame and replace MKQA dataset
        translated_df = pd.DataFrame.from_records(records)

        # Remove the rows with missing answers
        translated_df.dropna(subset="answer", inplace=True)

        # Set datatypes of integer columns
        translated_df.answer_start = translated_df.answer_start.astype(int)
        translated_df.answer_start_en = translated_df.answer_start_en.astype(int)

        # Return the translated dataset
        return translated_df

    def split_dataset(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Splits the dataset into train, validation and test sets.

        The dataset is split based on a stratification on whether an answer exists, to
        ensure that every split has roughly the same amount of questions without an
        answer.

        Args:
            dfs (Dict[str, pd.DataFrame]):
                The merged datasets from MKQA and NQ, with keys for each language.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]:
                The resulting datasets for each language and each split. The outer
                dictionary has the languages as keys and the inner dictionary has
                the splits ("train", "val" and "test") as keys.
        """
        # Add feature to the dataframes on whether an answer exists
        for language, df in dfs.items():
            dfs[language]["has_answer"] = df.answer.str.len() > 0

        # Get the example IDs that all languages have in common
        common_example_ids = np.asarray(
            list(
                set.intersection(*[set(df.example_id.tolist()) for df in dfs.values()])
            )
        )

        # Get matrix of shape (num_languages, num_common_example_ids) indicating
        # whether an example has an answer in each language
        has_answer_matrix = np.stack(
            [
                df.set_index("example_id").loc[common_example_ids, "has_answer"]
                for df in dfs.values()
            ],
            axis=0,
        )

        # Compute an array of shape (num_common_example_ids,) indicating, for each
        # sample, whether all languages agree on whether it has an answer
        has_same_answer = np.array(
            [
                np.unique(has_answer_matrix[:, col_idx]).size == 1
                for col_idx in range(has_answer_matrix.shape[1])
            ]
        )

        # Get the common example IDs where all languages agree on whether an answer
        # exists
        common_example_ids = common_example_ids[has_same_answer]

        # Extract a combined validation and test set, stratified on whether an answer
        # exists
        _, val_test_ids = train_test_split(
            common_example_ids,
            test_size=1000,
            stratify=dfs[self.languages[0]].loc[common_example_ids].has_answer,
        )

        # Extract the validation and test sets, again stratified on whether an answer
        # exists
        val_ids, test_ids = train_test_split(
            val_test_ids,
            test_size=500,
            stratify=dfs[self.languages[0]].loc[val_test_ids].has_answer,
        )

        # Extract the training IDs as the remaining IDs
        train_ids = {
            language: set(df.example_id.tolist())
            .difference(val_ids)
            .difference(test_ids)
            for language, df in dfs.items()
        }
        breakpoint()

        # Split the datasets into train, validation and test sets and return them
        return {
            language: dict(
                train=df[df.example_id.isin(train_ids[language])],
                val=df[df.example_id.isin(val_ids)],
                test=df[df.example_id.isin(test_ids)],
            )
            for language, df in dfs.items()
        }

    def _translate_context(self, example: pd.Series, language: str) -> dict:
        """Translate the English context to the target language.

        Args:
            example (pd.Series):
                The MKQA example with an English context that needs to be translated.
            language (str):
                The language to translate the context to.

        Returns:
            dict:
                The example with the translated context as the 'context' attribute and
                answer as the 'answer' and 'answer_start' attributes. The answer is
                None if the answer, or variants thereof, does not appear in the
                translated context.
        """
        # Translate the English context
        example["context"] = self.translators[language](
            example.context_en, target_lang=language
        )

        # Append the English title to both the English context and the translated
        # context, if it does not already appear in the context
        if example.title_en not in example.context:
            example["context"] = f"{example.title_en}\n{example.context}"
        if example.title_en not in example.context_en:
            example["context_en"] = f"{example.title_en}\n{example.context_en}"

        # If the example does not have an MKQA answer then we simply use the translated
        # context as the new context and set the starting index to be -1
        if example.answer == "":
            example["answer_start"] = -1
            example["answer_start_en"] = -1

        # Otherwise, attempt to extract the answer from the translated context
        else:
            answer_dict = extract_answer(
                answer=example.answer,
                answer_en=example.answer_en,
                context=example.context,
                language=language,
                translator=self.translators[language],
            )

            # If no answer could be extracted then set the answer and answer_start to
            # None
            if answer_dict is None:
                example["answer"] = None
                example["answer_start"] = None

            # Otherwise, we set the answer candidate appearing in the translated
            # context as the answer, and the translated context as the context
            else:

                # Sanity check that the found answer actually appears in the translated
                # context, at the correct position
                ctx_answer_start = answer_dict["answer_start"]
                ctx_answer_end = answer_dict["answer_start"] + len(
                    answer_dict["answer"]
                )
                ctx_answer = example.context[ctx_answer_start:ctx_answer_end]
                assert answer_dict["answer"] == ctx_answer

                # Store the found answer and its starting index in the example
                example["answer"] = answer_dict["answer"]
                example["answer_start"] = answer_dict["answer_start"]

                # If the English answer did not appear in the English context, then
                # check if the answer appears in the English context instead, and use
                # that as the English answer and starting index
                if example.answer_start_en == -1:
                    answer_en_dict = extract_answer(
                        answer=example.answer,
                        answer_en=None,
                        context=example.context_en,
                        language="en",
                        translator=self.translators[language],
                    )
                    if answer_en_dict is not None:
                        example["answer_en"] = answer_en_dict["answer"]
                        example["answer_start_en"] = answer_en_dict["answer_start"]

        # Add the example_id to the example
        example["example_id"] = example.name

        # Return the example as a dictionary
        return example.to_dict()
