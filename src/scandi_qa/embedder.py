"""Class which embeds texts and computes cosine similarities between them."""

from typing import List, Union

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

    def similarity(
        self, text1: str, text2: Union[List[str], str]
    ) -> Union[float, List[float]]:
        """Compute the cosine similarity between two texts.

        Args:
            text1 (str):
                The first text.
            text2 (str or list of str):
                The second text, which can also be a list of strings. If the latter
                then the cosine similarity is computed between the embeddings of
                the first text and all the embeddings of the texts in the list

        Returns:
            float or list of float:
                The cosine similarity between the two texts, or a list of cosine
                similarities if the second text is a list of strings.
        """
        # Ensure that `text2` is a list
        if isinstance(text2, str):
            text2 = [text2]

        # Embed the texts
        text1_embedding = self.embed(text1)
        text2_embeddings = [self.embed(doc) for doc in text2]

        # Normalise the embeddings
        text1_embedding = text1_embedding / np.linalg.norm(text1_embedding)
        text2_embeddings = [emb / np.linalg.norm(emb) for emb in text2_embeddings]

        # Compute the cosine similarity
        similarities = [np.dot(text1_embedding, emb) for emb in text2_embeddings]

        # If `text2` was a single string then return the single similarity
        if len(similarities) == 1:
            return similarities[0]
        else:
            return similarities
