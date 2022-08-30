"""Cleaning of contexts and questions."""

import re
import unicodedata
from typing import Optional


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


def clean_context_or_answer(context_or_answer: Optional[str]) -> str:
    """Clean the context or answer.

    Args:
        context_or_answer (str or None):
            The context or answer to clean.

    Returns:
        str:
            The cleaned context or answer.
    """
    # If the document is None or empty then return an empty string
    if context_or_answer is None or context_or_answer == "":
        return ""

    # NFKC normalise the context or answer
    cleaned = unicodedata.normalize("NFKC", context_or_answer)

    # Remove the Wikipedia reference tags from the context
    cleaned = re.sub(r"\[([0-9]+|citation needed)\]", "", cleaned)

    # Strip context of trailing whitespace and newlines
    cleaned = cleaned.strip().strip("\n")

    # Return the cleaned context
    return cleaned
