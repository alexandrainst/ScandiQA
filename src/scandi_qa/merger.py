"""Class that filters the Natural Questions dataset."""

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from datasets.arrow_dataset import Example
from tqdm.auto import tqdm

from .answer_extraction import extract_answer, generate_answer_candidates
from .cleaning import clean_answer, clean_context, clean_question
from .embedder import Embedder


class Merger:
    """Class that merges the Natural Questions dataset with the MKQA dataset.

    Args:
        language (str, optional):
            The desired MKQA language. Must be either "en", "da", "sv" or "no".
            Defaults to "da".
        cache_dir (str, optional):
            The directory to cache the merged dataset. Defaults to
            "~/.cache/huggingface/datasets".

    Attributes:
        language (str): The desired language.
        cache_dir (str): The cache directory.
        mkqa (pd.DataFrame): The MKQA dataset.
        nq (Hugging Face Dataset): The Natural Questions dataset.
        embedder (Embedder): The embedder used to embed the questions and contexts.
    """

    def __init__(
        self, language: str = "da", cache_dir: str = "~/.cache/huggingface/datasets"
    ):
        self.language = language
        self.cache_dir = cache_dir
        self.mkqa = self.build_mkqa()
        self.nq = load_dataset(
            "natural_questions", split="train", cache_dir=self.cache_dir
        )
        self.embedder = Embedder()

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
        mkqa.question = mkqa.question.map(clean_question)

        # Clean the answers
        mkqa.answer = mkqa.answer.map(clean_answer)
        mkqa.answer_en = mkqa.answer_en.map(clean_answer)

        # Remove the 'queries' and 'answers' columns
        mkqa.drop(columns=["query", "queries", "answers"], inplace=True)

        # Set the index to the example ID
        mkqa = mkqa.astype(dict(example_id="int64"))
        mkqa.set_index("example_id", inplace=True)

        # Return the processed MKQA dataset
        return mkqa

    def merge(self) -> pd.DataFrame:
        """Merges the Natural Questions dataset with the MKQA dataset.

        Returns:
            Pandas DataFrame:
                The merged dataset.
        """
        # Create copy of the MKQA dataset
        dataset = self.mkqa.copy()

        # Create dictionaries that stores the English titles and contexts of the
        # examples
        titles = dict()
        contexts = dict()
        answer_ens = dict()
        answer_start_ens = dict()

        # Iterate over the examples in the natural questions dataset
        for example in tqdm(self.nq, desc="Processing examples"):

            # Get the example ID
            example_id = int(example["id"])

            # Check if the example is a yes/no answer, as we do not want to include
            # those
            yes_no_answer = example["annotations"]["yes_no_answer"][0] == 1

            # Check if the example ID is in the MKQA dataset
            if example_id in dataset.index and not yes_no_answer:

                # Process the example
                example = self._process_nq_example(example)

                # Add the context to the dictionary
                titles[example_id] = example["title_en"]
                contexts[example_id] = example["context_en"]
                answer_ens[example_id] = example["answer_en"]
                answer_start_ens[example_id] = example["answer_start_en"]

        # Add the titles and contexts as columns in the MKQA dataset
        dataset["title_en"] = dataset.index.map(titles)
        dataset["context_en"] = dataset.index.map(contexts)
        dataset["answer_en"] = dataset.index.map(answer_ens)
        dataset["answer_start_en"] = dataset.index.map(answer_start_ens)

        # Remove the rows with missing contexts
        dataset.dropna(subset="context_en", inplace=True)

        # Return the resulting dataset
        return dataset

    def _process_nq_example(self, example: Example) -> Example:
        """Processes an example from the NQ dataset.

        Args:
            example (Example):
                The example to process.

        Returns:
            Example:
                The processed example, with keys 'example_id', 'title_en' and
                'context_en'. The 'context_en' feature is set to None if either there
                are no <p>, <span> or <table> tags with more than 200 characters in the
                HTML, or that the given example does not have a long answer in NQ, has
                an answer in MKQA, but the answer, or variants thereof, does not appear
                in any <p>, <span> or <table> tags.
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
        answer = self.mkqa.loc[example_id, "answer"]
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

            # Extract all the paragraphs from the HTML context. These are all the <p>,
            # <span> and <table> tags in the HTML context which contain more than 200
            # characters
            soup = BeautifulSoup(html_bytes, "html.parser")
            context_candidates = [
                tag.get_text().strip("\n")
                for tag_name in ["p", "span", "table"]
                for tag in soup.find_all(tag_name)
                if len(tag.get_text()) > 200
            ]

            # If no candidate contexts were found then we discard the example by
            # setting the context to None
            if len(context_candidates) == 0:
                context_en = None
                answer_en = None
                answer_start_en = None

            # Otherwise, we find the context with the highest cosine similarity with
            # the question
            else:
                # Compute the similarity between the question and all the paragraphs
                similarities = self.embedder.similarities(question, context_candidates)

                # Get the paragraph with the largest similarity
                best_idx = similarities.index(max(similarities))  # type: ignore
                context_en = context_candidates[best_idx]

        # Otherwise, if there is no long answer but there *is* an answer in MKQA, we
        # extract all the answer candidates from the English version of the MKQA
        # answer, and use the <p>, <span> or <table> tag that contains one of the
        # candidate answers and has the largest cosine similarity with the question
        else:

            # Get list of answer candidates
            answer_candidates = generate_answer_candidates(
                answer=answer_en, language="en"
            )

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
                answer_en = None
                answer_start_en = None

            # Otherwise, we choose the context candidate with the highest cosine
            # similarity with the question
            else:

                # Get the question
                question = example["question"]["text"]

                # Compute the similarity between the question and all the candidate
                # contexts
                similarities = self.embedder.similarities(question, context_candidates)

                # Get the candidate context with the largest similarity
                best_idx = similarities.index(max(similarities))  # type: ignore
                context_en = context_candidates[best_idx]

        # Clean the context if it exists
        if context_en is not None:
            context_en = clean_context(context_en)

            # Extract the answer from the cleaned context
            answer_dict = extract_answer(
                answer=answer_en, context=context_en, language=self.language
            )

            # If no answer was found then try searching for the language-specific
            # answer in the English context instead
            if answer_dict is None:
                answer_dict = extract_answer(
                    answer=answer, context=context_en, language=self.language
                )

            # If the answer was still not found then we set the answer to the empty
            # string
            if answer_dict is None:
                answer_en = ""
                answer_start_en = -1

            # Otherwise, we set the answer to the answer found in the cleaned context
            else:
                answer_en = answer_dict["answer"]
                answer_start_en = answer_dict["answer_start"]

        # Add the example ID, title and context to the example
        example["example_id"] = example_id
        example["title_en"] = title
        example["context_en"] = context_en
        example["answer_en"] = answer_en
        example["answer_start_en"] = answer_start_en

        # Remove the 'id', 'document', 'question' and 'annotations' keys
        example.pop("id")
        example.pop("document")
        example.pop("question")
        example.pop("annotations")

        # Return the processed example
        return example
