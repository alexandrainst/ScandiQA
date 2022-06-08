"""Loading and processing of data."""

import re

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
            The language of the dataset. Needs to be a language present in MKQA.
            Defaults to 'da'.
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
        # Extract the title
        title = example["document"]["title"]

        # Add the title to the example
        example["title_en"] = title

        # Extract the document bytes of the raw HTML context
        html_bytes = example["document"]["html"].encode("utf-8")

        # Extract the byte indices of the long answer
        long_answer_dict = example["annotations"]["long_answer"][0]
        long_answer_start = long_answer_dict["start_byte"]
        long_answer_end = long_answer_dict["end_byte"]

        # If the long answer does not exist, then we want to set the context to the
        # <p> tag which has the largest semantic similarity to the question
        if long_answer_start == -1 or long_answer_end == -1:

            # Get the question
            question = example["question"]["text"]

            # Embed the question
            question_emb = self.sbert.encode(question)

            # Extract all the paragraphs from the HTML context. These are all the <p>
            # tags in the HTML context which contain more than 10 characters
            soup = BeautifulSoup(html_bytes, "html.parser")
            paragraphs = [
                p.get_text().strip("\n")
                for p in soup.find_all("p")
                if len(p.get_text()) > 10
            ]

            # Embed all the paragraphs
            paragraphs_emb = [self.sbert.encode(p) for p in paragraphs]

            # Compute the similarity between the question and all the paragraphs
            similarities = [self.similarity(question_emb, p) for p in paragraphs_emb]

            # Get the paragraph with the largest similarity
            context_en = paragraphs[similarities.index(max(similarities))]

            # Set the answer start to -1
            answer_start = -1

        # Otherwise, we want to use the long answer as the context
        else:

            # Extract the long answer as HTML
            long_answer_html = html_bytes[long_answer_start:long_answer_end]

            # Parse the HTML to get the long answer as plain text
            long_answer = BeautifulSoup(long_answer_html, "html.parser").get_text()

            # Remove the Wikipedia reference tags from the long answer
            context_en = re.sub(r"\[[0-9]+\]", "", long_answer).strip()

            # Get the answer along with the byte indices
            answer_dict = example["annotations"]["short_answers"][0]
            answer = answer_dict["text"]
            answer_start = answer_dict["start_byte"]
            answer_end = answer_dict["end_byte"]

            # Double-check that the start and stop byte indices of the answer is indeed
            # the answer
            assert answer == html_bytes[answer_start:answer_end].decode("utf-8")

            # Check how many times the answer appears in the context
            answer_count = context_en.count(answer)

            # If the answer appears only once, then we identify the start index by simply
            # using the `index` method
            if answer_count == 1:
                answer_start = context_en.index(answer)

            # Otherwise, we need to find what occurence our desired answer is in the
            # HTML context, and find the corresponding start index in the parsed context
            else:
                # Find all start indices of the answer in the HTML context
                answer_html_idxs = [s.start() for s in re.finditer(answer, html_bytes)]

                # Find the occurence of the desired answer in the HTML context
                answer_occurence = answer_html_idxs.index(answer_start)

                # Find all start indices of the answer in the parsed context
                answer_parsed_idxs = [
                    s.start() for s in re.finditer(answer, long_answer)
                ]

                # Extract the start index of the desired answer in the parsed context
                answer_start = answer_parsed_idxs[answer_occurence]

        # Double-check that the context is not empty
        assert len(context_en) > 0

        # Add the context to the example
        example["context_en"] = context_en

        # Add the answer start index to the example
        example["answer_start_en"] = answer_start

        # Rename the 'id' to 'example_id'
        example["example_id"] = example["id"]

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

        # Remove the 'queries' and 'answers' columns
        mkqa.drop(columns=["query", "queries", "answers"], inplace=True)

        # Set the index to the example ID
        mkqa = mkqa.astype(dict(example_id="int64"))
        mkqa.set_index("example_id", inplace=True)

        # Return the processed MKQA dataset
        return mkqa

    def push_to_hub(self):
        """Pushes the dataset to the Hugging Face Hub."""
        mkqa_dataset = Dataset.from_pandas(self.mkqa)
        mkqa_dataset.push_to_hub(f"mkqa_{self.language}")
        return self


if __name__ == "__main__":
    # cache_dir = "/mnt/data_4tb/dan/.cache/huggingface"
    for language in ["da", "sv", "no"]:
        dataset = ScandiQADataset(language=language)
        dataset.add_english_contexts()
        dataset.push_to_hub()
