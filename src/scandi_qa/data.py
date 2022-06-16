"""Loading and processing of data."""

import re
import unicodedata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from datasets.arrow_dataset import Example
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .translation import DeepLTranslator
from .utils import (
    DANISH_NUMERALS,
    ENGLISH_NUMERALS,
    NORWEGIAN_NUMERALS,
    SWEDISH_NUMERALS,
)


class ScandiQADataset:
    """ScandiQA dataset class.

    Args:
        language (str, optional):
            The language of the dataset. Can be "da", 'sv' or 'no. Defaults to 'da'.
        cache_dir (str, optional):
            The directory to cache the dataset. Defaults to
            '~/.cache/huggingface/datasets'.

    Attributes:
        language (str):
            The language of the dataset.
        mkqa (Pandas DataFrame):
            The MKQA dataset for the given language.
        nq (Dataset):
            The Google Natural Questions dataset.

    Raises:
        ValueError:
            If the language is not supported.
    """

    def __init__(
        self, language: str = "da", cache_dir: str = "~/.cache/huggingface/datasets"
    ):
        # If the language is not supported, raise an error
        if language not in ["da", "sv", "no"]:
            raise ValueError(f"Language '{language}' not supported")

        self.language = language
        self.cache_dir = cache_dir
        self.mkqa = self.build_mkqa()
        self.nq = load_dataset(
            "natural_questions", split="train", cache_dir=self.cache_dir
        )
        self.sbert = SentenceTransformer("all-mpnet-base-v2")
        self.translator = DeepLTranslator()

    def build(self):
        """Builds the dataset and pushes it to the Hugging Face Hub."""

        # Add English contexts
        self.add_english_contexts()

        # Translate the English contexts
        # self.translate_contexts()

        # Compute start indices of the answers
        # self.add_answer_indices()

        # Push to the Hub
        # self.push_to_hub()

        return self

    def add_english_contexts(self):
        """Adds English contexts to the MKQA dataset.

        This finds, for each example in the MKQA dataset, the corresponding
        example in the Google Natural Questions dataset, and adds the long answer from
        the latter as context for the former.
        """
        # Create dictionaries that stores the English titles and contexts of the
        # examples
        titles = dict()
        contexts = dict()

        # Iterate over the examples in the natural questions dataset
        for example in tqdm(self.nq, desc="Processing examples"):

            # Get the example ID
            example_id = int(example["id"])

            # Check if the example is a yes/no answer, as we do not want to include
            # those
            yes_no_answer = example["annotations"]["yes_no_answer"][0] == 1

            # Check if the example ID is in the MKQA dataset
            if example_id in self.mkqa.index and not yes_no_answer:

                # Process the example
                example = self._process_nq_example(example)

                # Add the context to the dictionary
                titles[example_id] = example["title_en"]
                contexts[example_id] = example["context_en"]

        # Add the titles and contexts as columns in the MKQA dataset
        self.mkqa["title_en"] = self.mkqa.index.map(titles)
        self.mkqa["context_en"] = self.mkqa.index.map(contexts)

        breakpoint()

        # Remove the rows with missing contexts
        self.mkqa.dropna(subset=["title_en", "context_en"], inplace=True)

        return self

    def translate_contexts(self):
        """Translates the English contexts of the MKQA dataset."""

        # Set up progress bar
        desc = "Translating contexts"
        with tqdm(self.mkqa.iterrows(), total=len(self.mkqa), desc=desc) as pbar:

            # Translate all the English contexts
            records = [self._translate_context(example) for _, example in pbar]

            # Convert to a Pandas DataFrame and replace MKQA dataset
            self.mkqa = pd.DataFrame.from_records(records)

            # Remove the rows with missing contexts and/or answers
            self.mkqa.dropna(subset=["answer", "context"], inplace=True)

    def add_answer_indices(self):
        """Adds the start indices of the answers to the MKQA dataset."""

        # Set up progress bar
        desc = "Adding answer indices"
        with tqdm(self.mkqa.iterrows(), total=len(self.mkqa), desc=desc) as pbar:

            # Translate all the English contexts
            records = [self._add_answer_index(example) for _, example in pbar]

            # Convert to a Pandas DataFrame and replace MKQA dataset
            self.mkqa = pd.DataFrame.from_records(records)

    def push_to_hub(self):
        """Pushes the dataset to the Hugging Face Hub."""
        # Convert to a Hugging Face Dataset
        mkqa_dataset = Dataset.from_pandas(self.mkqa)

        # Push the dataset to the Hub
        mkqa_dataset.push_to_hub(f"mkqa_{self.language}")

        return self

    def build_mkqa(self) -> pd.DataFrame:
        """Builds the MKQA dataset for the given language.

        Returns:
            Pandas DataFrame:
                The MKQA dataset for the given language.
        """
        # Load the raw MKQA dataset
        mkqa = load_dataset("mkqa", split="train", cache_dir=self.cache_dir).to_pandas()

        # Get the language-specific queries and answers
        mkqa["question"] = mkqa.queries.map(lambda dct: dct[self.language])
        mkqa["answer"] = mkqa.answers.map(lambda dct: dct[self.language][0]["text"])
        mkqa["answer_en"] = mkqa.answers.map(lambda dct: dct["en"][0]["text"])

        # Clean the questions
        mkqa.question = mkqa.question.map(self._clean_question)

        # Remove the 'queries' and 'answers' columns
        mkqa.drop(columns=["query", "queries", "answers"], inplace=True)

        # Set the index to the example ID
        mkqa = mkqa.astype(dict(example_id="int64"))
        mkqa.set_index("example_id", inplace=True)

        # Return the processed MKQA dataset
        return mkqa

    @staticmethod
    def _similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute the cosine similarity between two embeddings.

        Args:
            emb1 (np.ndarray):
                The first embedding.
            emb2 (np.ndarray):
                The second embedding.

        Returns:
            float:
                The cosine similarity between the two embeddings.
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    @staticmethod
    def _clean_question(question: str) -> str:
        """Clean the question of an MKQA example.

        Args:
            question (str):
                The question to clean.

        Returns:
            str:
                The cleaned question.
        """
        # Remove multiple whitespace
        cleaned_question = re.sub(r"\s+", " ", question)

        # Ensure that the first character of the question is capitalised
        cleaned_question = cleaned_question.capitalize()

        # Strip the question of any leading or trailing whitespace
        cleaned_question = cleaned_question.strip()

        # Add a question mark at the end of the question if it is missing
        if not cleaned_question.endswith("?"):
            cleaned_question += "?"

        # Return the cleaned question
        return cleaned_question

    @staticmethod
    def _clean_context(context: str) -> str:
        """Clean the context of an Natural Questions example.

        Args:
            context (str):
                The context to clean.

        Returns:
            str:
                The cleaned context.
        """
        # NFKC normalise the context
        cleaned_context = unicodedata.normalize("NFKC", context)

        # Remove the Wikipedia reference tags from the context
        cleaned_context = re.sub(r"\[([0-9]+|citation needed)\]", "", cleaned_context)

        # Strip context of trailing whitespace and newlines
        cleaned_context = cleaned_context.strip().strip("\n")

        # Check that the cleaned context is not empty
        assert len(cleaned_context) > 0

        # Return the cleaned context
        return cleaned_context

    def _process_nq_example(self, example: Example) -> Example:
        """Processes an example from the NQ dataset.

        Args:
            example (Example):
                The example to process.

        Returns:
            Example:
                The processed example, with keys 'example_id', 'title_en' and
                'context_en'. The 'context_en' feature is set to None if either there
                are no <p>, <span> or <table> tags in the HTML, or that the given
                example does not have a long answer in NQ, has an answer in MKQA, but
                the answer, or variants thereof, does not appear in any <p>, <span> or
                <table> tags.
        """
        # Extract the example ID
        example_id = int(example["id"])

        # Extract the title
        title = example["document"]["title"]

        # Extract the document bytes of the raw HTML context
        html = example["document"]["html"]
        html_bytes = html.encode("utf-8")

        # Extract the byte indices of the long answer
        long_answer_dict = example["annotations"]["long_answer"][0]
        long_answer_start = long_answer_dict["start_byte"]
        long_answer_end = long_answer_dict["end_byte"]

        # Define variable determining whether the long answer exists
        has_long_answer = long_answer_start != -1 and long_answer_end != -1

        # Define variable on whether the MKQA dataset contains an answer for the
        # example
        answer_en = self.mkqa.loc[example_id, "answer_en"]
        mkqa_has_answer = answer_en is not None

        # If the long answer exists then use this as the context
        if has_long_answer:

            # Extract the long answer as HTML
            long_answer_html = html_bytes[long_answer_start:long_answer_end]

            # Parse the HTML to get the long answer as plain text
            context_en = BeautifulSoup(long_answer_html, "html.parser").get_text()

        # Otherwise, if there is neither a long answer nor an answer in MKQA then use
        # the <p> tag that has the largest cosine similarity with the question as the
        # context
        elif not mkqa_has_answer:

            # Get the question
            question = example["question"]["text"]

            # Embed the question
            question_emb = self.sbert.encode(question)

            # Extract all the paragraphs from the HTML context. These are all the <p>,
            # <span> and <table> tags in the HTML context which contain more than 10
            # characters
            soup = BeautifulSoup(html_bytes, "html.parser")
            context_candidates = [
                tag.get_text().strip("\n")
                for tag_name in ["p", "span", "table"]
                for tag in soup.find_all(tag_name)
                if len(tag.get_text()) > 10
            ]

            # If no candidate contexts were found then we discard the example by
            # setting the context to None
            if len(context_candidates) == 0:
                context_en = None

            # Otherwise, we find the context with the highest cosine similarity with
            # the question
            else:
                # Embed all the paragraphs
                candidate_embs = [self.sbert.encode(ctx) for ctx in context_candidates]

                # Compute the similarity between the question and all the paragraphs
                similarities = [
                    self._similarity(question_emb, p) for p in candidate_embs
                ]

                # Get the paragraph with the largest similarity
                context_en = context_candidates[similarities.index(max(similarities))]

        # Otherwise, if there is no long answer but there *is* an answer in MKQA, we
        # extract all the answer candidates from the English version of the MKQA
        # answer, and use the <p>, <span> or <table> tag that contains one of the
        # candidate answers and has the largest cosine similarity with the question
        else:

            # Create singleton list of answer candidates
            answer_candidates = [answer_en]

            # If the answer looks like an integer, then add the corresponding
            # written form of the integer to the answer candidates
            if re.match(r"^[0-9]+(\.0)?$", answer_en) is not None:

                # Extract the integer
                integer = int(re.sub(r"\.0", "", answer_en))

                # Add the written form of the integer to the answer candidates
                if integer >= 0 and integer <= 20:
                    answer_candidates.extend(ENGLISH_NUMERALS[integer])

            # Extract all the <p>, <span> and <table> tags in the HTML context which
            # contain more than 200 characters and which contain a candidate answer
            soup = BeautifulSoup(html_bytes, "html.parser")
            context_candidates = [
                tag.get_text().strip("\n")
                for tag_name in ["p", "span", "table"]
                for tag in soup.find_all(tag_name)
                if len(tag.get_text()) > 200
                and any(
                    candidate.lower() in tag.get_text().lower()
                    for candidate in answer_candidates
                )
            ]

            # If none of the candidate contexts were found then we discard the
            # example by setting the context to None
            if len(context_candidates) == 0:
                context_en = None

            # Otherwise, we choose the context candidate with the highest cosine
            # similarity with the question
            else:

                # Get the question
                question = example["question"]["text"]

                # Embed the question
                question_emb = self.sbert.encode(question)

                # Embed all the candidate contexts
                candidate_embs = [self.sbert.encode(ctx) for ctx in context_candidates]

                # Compute the similarity between the question and all the candidate
                # contexts
                similarities = [
                    self._similarity(question_emb, p) for p in candidate_embs
                ]

                # Get the candidate context with the largest similarity
                context_en = context_candidates[similarities.index(max(similarities))]

        # Clean the context if it exists
        if context_en is not None:
            context_en = self._clean_context(context_en)

        # Add the example ID, title and context to the example
        example["example_id"] = example_id
        example["title_en"] = title
        example["context_en"] = context_en

        # Remove the 'id', 'document', 'question' and 'annotations' keys
        example.pop("id")
        example.pop("document")
        example.pop("question")
        example.pop("annotations")

        # Return the processed example
        return example

    def _translate_context(self, example: pd.Series) -> dict:
        """Translate the English context to the target language.

        Args:
            example (pd.Series):
                The MKQA example with an English context that needs to be translated.

        Returns:
            dict:
                The example with the translated context.
        """
        # Translate the English context
        example["context"] = self.translator(
            example.context_en, target_lang=self.language
        )

        # If the example does not have an answer then we simply use the translated
        # context as the new context
        if example.answer is None:
            return example.to_dict()

        # Create singleton list of answer candidates
        answer_candidates = [example.answer]

        # If the answer looks like an integer, then add the corresponding
        # written form of the integer to the answer candidates
        if re.match(r"^[0-9]+(\.0)?$", example.answer) is not None:

            # Extract the integer
            integer = int(re.sub(r"\.0", "", example.answer))

            # Get the written form of the integer if it is between 0 and 20, inclusive
            if integer >= 0 and integer <= 20:

                # Add the written form of the integer to the answer candidates
                if self.language == "da":
                    answer_candidates.extend(DANISH_NUMERALS[integer])
                elif self.language == "sv":
                    answer_candidates.extend(SWEDISH_NUMERALS[integer])
                else:
                    answer_candidates.extend(NORWEGIAN_NUMERALS[integer])

        # Create variable storing whether any of the answer candidates appear
        # in the translated context
        has_answer = any(
            candidate.lower() in example.context.lower()
            for candidate in answer_candidates
        )

        # If none of the answer candidates appear in the translated context
        # then we discard the example by setting the context to None
        if not has_answer:
            example["context"] = None
            example["answer"] = None

        # Otherwise, we set the answer candidate appearing in the translated
        # context as the answer, and the translated context as the context
        else:

            # Extract the answer candidate appearing in the translated context
            answer = next(
                candidate
                for candidate in answer_candidates
                if candidate.lower() in example.context.lower()
            )

            # Get the index at which the answer appears in the context
            answer_idx = example.context.lower().index(answer.lower())

            # Use the index to extract the answer with correct casing from the
            # context
            example["answer"] = example.context[answer_idx : answer_idx + len(answer)]

        # Return the example as a dictionary
        return example.to_dict()

    def _add_answer_index(self, example: pd.Series) -> dict:
        """Adds the answer starting index to an example.

        Args:
            example (pd.Series):
                The MKQA example with a translated context.

        Returns:
            dict:
                The example with an answer starting index.
        """
        # Start by setting the answer starting index to -1, which is the default until
        # we find an alternative index
        example["answer_start"] = -1
        example["answer_start_en"] = -1

        # Create variable containing the contexts and answers
        answer_contexts = [
            ("answer_start", example.answer, example.context),
            ("answer_start_en", example.answer_en, example.context_en),
        ]

        # If there is an answer and it appears in the context, then we use the
        # first index of the answer in the context as the answer starting index
        for new_col_name, answer, context in answer_contexts:
            if answer is not None and answer in context:
                example[new_col_name] = context.index(answer)

        # If the English answer did not appear in the English context, then check if
        # the answer appears in the English context instead, and use that as the
        # English starting index
        if example.answer_start_en == -1 and example.answer in example.context_en:
            example["answer_start_en"] = example.context_en.index(example.answer)
            example["answer_en"] = example.answer

        return example.to_dict()


if __name__ == "__main__":
    # cache_dir = "/mnt/data_4tb/dan/.cache/huggingface"
    languages = ["da"]  # ["da", "sv"]
    for language in languages:
        dataset = ScandiQADataset(language=language)
        dataset.build()
