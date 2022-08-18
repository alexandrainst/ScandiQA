"""Class which embeds texts and computes cosine similarities between them."""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """Class which embeds texts and computes cosine similarities between them.

    Args:
        model_id (str, optional):
            The model ID to use. Defaults to "all-mpnet-base-v2".

    Attributes:
        device (str): The device to use.
        model (SentenceTransformer): The sentence transformer model.
    """

    def __init__(self, model_id: str = "all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert = SentenceTransformer(model_id, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        """Embed a text.

        Args:
            text (str):
                The text to embed.

        Returns:
            Numpy array:
                The embedding of the text.
        """
        return self.sbert.encode(text, convert_to_numpy=True)

    def similarities(self, doc: str, other_docs: List[str]) -> List[float]:
        """Compute the cosine similarity between two texts.

        Args:
            doc (str):
                The document to compare to other documents.
            other_docs (list of str):
                The list of documents to compare to `doc`.

        Returns:
            list of float:
                The cosine similarity between the two texts, or a list of cosine
                similarities if the second text is a list of strings.
        """
        # Embed the texts
        doc_embedding = self.embed(doc)
        other_docs_embeddings = [self.embed(doc) for doc in other_docs]

        # Normalise the embeddings
        doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)
        other_docs_embeddings = [
            emb / np.linalg.norm(emb) for emb in other_docs_embeddings
        ]

        # Compute the cosine similarity
        similarities = [np.dot(doc_embedding, emb) for emb in other_docs_embeddings]

        # If the second string was a single string then return the single similarity
        return similarities
