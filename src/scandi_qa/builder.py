"""Loading and processing of data."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
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
        val_size (int, optional):
            The size of the validation set. Defaults to 750.
        test_size (int, optional):
            The size of the test set. Defaults to 750.
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
        self,
        language: str = "da",
        val_size: int = 750,
        test_size: int = 750,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        # If the language is not supported, raise an error
        if language not in MKQA_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. We only support "
                f"{', '.join(MKQA_LANGUAGES)}"
            )

        self.language = language
        self.val_size = val_size
        self.test_size = test_size
        self.cache_dir = cache_dir

        # Set up translator, depending on the language
        self.translator: Translator
        if language in DEEPL_LANGUAGES:
            self.translator = DeepLTranslator()
        else:
            self.translator = GoogleTranslator()

        self.merger = Merger(
            translator=self.translator,
            language=self.language,
            cache_dir=self.cache_dir,
        )

    def build(self) -> pd.DataFrame:
        """Builds the dataset and pushes it to the Hugging Face Hub.

        Returns:
            Pandas DataFrame:
                The resulting dataset for the given language.
        """
        # Set up the path to the merged dataset
        merged_path = Path("data") / "processed" / f"merged_{self.language}.parquet"

        # Merge the MKQA and NQ datasets if they haven't been merged yet
        if not merged_path.exists():
            df = self.merger.merge()
            df.to_parquet(merged_path)

        # Otherwise load the merged dataset
        else:
            df = pd.read_parquet(merged_path)

        # Translate the English contexts
        df = self.translate_contexts(df)

        # Split into train, validation and test sets
        dataset_dict = self.split_dataset(df)

        # Push to the Hub
        dataset_dict.push_to_hub(f"scandiqa-{self.language}")

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

    def split_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """Splits the dataset into train, validation and test sets.

        The dataset is split based on a stratification on whether an answer exists, to
        ensure that every split has roughly the same amount of questions without an
        answer.

        Args:
            df (pd.DataFrame):
                The merged dataset from MKQA and NQ.

        Returns:
            DatasetDict:
                The dataset split into train, validation and test sets.
        """
        # Split the dataframe into one with and one without answers
        df_with_answer = df.query("answer != ''")
        df_without_answer = df.query("answer == ''")

        # Get the proportion of questions with an answer
        proportion_with_answer = len(df_with_answer) / len(df)
        proportion_without_answer = 1 - proportion_with_answer

        # Compute the number of val/test samples for answers/no answers
        val_test_size_with_answer = int(
            (self.val_size + self.test_size) * proportion_with_answer
        )
        val_test_size_without_answer = int(
            (self.val_size + self.test_size) * proportion_without_answer
        )

        # Split the dataframe with answers into a training split and a val/test split
        val_test_with_answer = df_with_answer.sample(n=val_test_size_with_answer)
        train_with_answer = df_with_answer.drop(val_test_with_answer.index)

        # Split the dataframe without answers into a training split and a val/test split
        val_test_without_answer = df_without_answer.sample(
            frac=val_test_size_without_answer
        )
        train_without_answer = df_without_answer.drop(val_test_without_answer.index)

        # Get the proportion of validation samples in the val/test split
        proportion_val = self.val_size / (self.val_size + self.test_size)

        # Split the val_test split with answers into a validation and a test split
        val_with_answer = val_test_with_answer.sample(frac=proportion_val)
        test_with_answer = val_test_with_answer.drop(val_with_answer.index)

        # Split the val_test split without answers into a validation and a test split
        val_without_answer = val_test_without_answer.sample(frac=proportion_val)
        test_without_answer = val_test_without_answer.drop(val_without_answer.index)

        # Concatenate the training, validation and test splits
        train = pd.concat([train_with_answer, train_without_answer])
        val = pd.concat([val_with_answer, val_without_answer])
        test = pd.concat([test_with_answer, test_without_answer])

        # Reset the indices of the splits
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        # Shuffle the training set
        train = train.sample(frac=1).reset_index(drop=True)

        # Convert the splits to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train)
        val_dataset = Dataset.from_pandas(val)
        test_dataset = Dataset.from_pandas(test)

        # Create a DatasetDict
        dataset_dict = DatasetDict(
            dict(
                train=train_dataset,
                val=val_dataset,
                test=test_dataset,
            )
        )

        # Return the DatasetDict
        return dataset_dict

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
                language=self.language,
                translator=self.translator,
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
                        translator=self.translator,
                    )
                    if answer_en_dict is not None:
                        example["answer_en"] = answer_en_dict["answer"]
                        example["answer_start_en"] = answer_en_dict["answer_start"]

        # Add the example_id to the example
        example["example_id"] = example.name

        # Return the example as a dictionary
        return example.to_dict()
