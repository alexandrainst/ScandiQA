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


def clean_answer(answer: Optional[str]) -> str:
    """Clean the answer of an MKQA example.

    Args:
        answer (str or None):
            The answer to clean.

    Returns:
        str:
            The cleaned answer.
    """
    # If the answer is None then set it to an empty string
    clean_answer = "" if answer is None else answer

    # Return the cleaned answer
    return clean_answer
