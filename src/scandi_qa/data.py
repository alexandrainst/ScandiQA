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
        answer_starts = dict()

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
                answer_starts[example_id] = example["answer_start_en"]

        # Add the titles and contexts as columns in the MKQA dataset
        self.mkqa["title_en"] = self.mkqa.index.map(titles)
        self.mkqa["context_en"] = self.mkqa.index.map(contexts)
        self.mkqa["answer_start_en"] = self.mkqa.index.map(answer_starts)

        # Remove the rows with missing contexts
        self.mkqa.dropna(
            subset=["title_en", "context_en", "answer_start_en"], inplace=True
        )

        # Cast the `answer_start_en` column as integer
        self.mkqa.answer_start_en = self.mkqa.answer_start_en.astype(int)

        return self

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
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
    def clean_question(question: str) -> str:
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
    def clean_context(context: str) -> str:
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
                The processed example, with keys 'example_id', 'title_en', 'context_en'
                and 'answer_start_en'.
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

        # Store variable on whether the MKQA dataset contains an answer for the
        # example
        answer = self.mkqa.loc[example_id, "answer"]
        has_answer = answer is not None

        # If the MKQA dataset does not contain an answer for the example, or if it does
        # contain one but the answer does not appear in the context, and the long
        # answer does not exist, then we want to set the context to the <p> tag which
        # has the largest semantic similarity to the question
        if (not has_answer or answer not in html) and (
            long_answer_start == -1 or long_answer_end == -1
        ):

            # Get the question
            question = example["question"]["text"]

            # Embed the question
            question_emb = self.sbert.encode(question)

            # Extract all the paragraphs from the HTML context. These are all the <p>
            # tags in the HTML context which contain more than 10 characters
            soup = BeautifulSoup(html_bytes, "html.parser")
            context_candidates = [
                tag.get_text().strip("\n")
                for tag_name in ["p", "span", "table"]
                for tag in soup.find_all(tag_name)
                if len(tag.get_text()) > 10
            ]

            # If no candidate contexts were found then we set the context to None,
            # which will mean that this example will be excluded from the dataset
            if len(context_candidates) == 0:
                context_en = None
                answer_start = None

            # Embed all the paragraphs
            candidate_embs = [self.sbert.encode(ctx) for ctx in context_candidates]

            # Compute the similarity between the question and all the paragraphs
            similarities = [self.similarity(question_emb, p) for p in candidate_embs]

            # Get the paragraph with the largest similarity
            context_en = context_candidates[similarities.index(max(similarities))]

            # Clean the context
            context_en = self.clean_context(context_en)

            # Set the answer start to -1
            answer_start = -1

        # Otherwise, if there *is* an answer in the MKQA dataset which also appears in
        # the context, but no long answer exists, then we want to extract the paragraph
        # from the HTML context that contains the answer.
        elif (
            has_answer
            and answer in html
            and (long_answer_start == -1 or long_answer_end == -1)
        ):

            # Extract all the paragraphs from the HTML context. These are all the <p>
            # tags in the HTML context
            soup = BeautifulSoup(html_bytes, "html.parser")
            context_candidates = [
                tag.get_text().strip("\n")
                for tag_name in ["p", "span", "table"]
                for tag in soup.find_all(tag_name)
                if answer in tag.get_text().strip("\n")
            ]

            # If no candidate contexts were found then we set the context to None,
            # which will mean that this example will be excluded from the dataset
            if len(context_candidates) == 0:
                context_en = None
                answer_start = None

            # Otherwise, we set the answer start to the index of the first paragraph
            # containing the answer
            else:
                # Clean the context
                context_en = self.clean_context(context_candidates[0])

                # Set the answer start to the index of the answer in the context
                answer_start = context_en.index(answer)

        # Otherwise, we want to use the long answer as the context
        else:

            # Extract the long answer as HTML
            long_answer_html = html_bytes[long_answer_start:long_answer_end]

            # Parse the HTML to get the long answer as plain text
            context_en = BeautifulSoup(long_answer_html, "html.parser").get_text()

            if len(context_en) == 0:
                breakpoint()

            # Clean the context
            context_en = self.clean_context(context_en)

            # Get the answer dictionary
            answer_dict = example["annotations"]["short_answers"][0]

            # If the answer does not exist or does not occur in the context then set
            # the answer start to -1
            if not has_answer or answer not in context_en:
                answer_start = -1

            # Otherwise, if there *is* an answer but no answer in the Natural Questions
            # dataset, then we set the answer start to the index of the answer in the
            # context
            elif len(answer_dict["text"]) == 0:
                answer_start = context_en.index(answer)

            # Otherwise, if there is an answer both in MKQA and Natural Questions, then
            # we extract the answer start index
            else:

                # Check how many times the answer appears in the context
                answer_count = context_en.count(answer)

                # If the answer appears only once, then we identify the start index by
                # simply using the `index` method
                if answer_count == 1:
                    answer_start = context_en.index(answer)

                # Otherwise, we need to find what occurence our desired answer is in
                # the HTML context, and find the corresponding start index in the
                # parsed context
                else:
                    # Get the Natural Questions answer start byte index
                    nq_answer_start = answer_dict["start_byte"][0]

                    # If the Natural Questions answer start byte index does indeed
                    # correspond to the MKQA answer then use this to extract the
                    # corresponding start index from the parsed HTML
                    if (
                        html_bytes[nq_answer_start : nq_answer_start + len(answer)]
                        == answer
                    ):

                        # Find all start indices of the answer in the HTML context
                        answer_html_idxs = [
                            s.start() for s in re.finditer(answer, html_bytes)
                        ]

                        # Find the occurence of the desired answer in the HTML context
                        answer_occurence = answer_html_idxs.index(nq_answer_start)

                        # Find all start indices of the answer in the parsed context
                        answer_parsed_idxs = [
                            s.start() for s in re.finditer(answer, context_en)
                        ]

                        # Extract the start index of the desired answer in the parsed
                        # context
                        answer_start = answer_parsed_idxs[answer_occurence]

                    # Otherwise, we set the answer start to the first occurence of the
                    # MKQA answer
                    else:
                        answer_start = context_en.index(answer)

        # Add the example ID, title, context and answer start index to the example
        example["example_id"] = example_id
        example["title_en"] = title
        example["context_en"] = context_en
        example["answer_start_en"] = answer_start

        # Remove the 'id', 'document', 'question' and 'annotations' keys
        example.pop("id")
        example.pop("document")
        example.pop("question")
        example.pop("annotations")

        # Return the processed example
        return example

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

        # Clean the questions
        mkqa.question = mkqa.question.map(self.clean_question)

        # Remove the 'queries' and 'answers' columns
        mkqa.drop(columns=["query", "queries", "answers"], inplace=True)

        # Set the index to the example ID
        mkqa = mkqa.astype(dict(example_id="int64"))
        mkqa.set_index("example_id", inplace=True)

        # Return the processed MKQA dataset
        return mkqa

    def push_to_hub(self):
        """Pushes the dataset to the Hugging Face Hub."""
        # Convert to a Hugging Face Dataset
        mkqa_dataset = Dataset.from_pandas(self.mkqa)

        # Push the dataset to the Hub
        mkqa_dataset.push_to_hub(f"mkqa_{self.language}")

        return self


if __name__ == "__main__":
    # cache_dir = "/mnt/data_4tb/dan/.cache/huggingface"
    for language in ["da", "sv", "no"]:
        dataset = ScandiQADataset(language=language)
        dataset.add_english_contexts()
        dataset.push_to_hub()
