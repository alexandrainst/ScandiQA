"""Extracting the answer from a context."""

import re
from typing import List, Optional, Union

from .utils import (
    DANISH_NUMERALS,
    ENGLISH_NUMERALS,
    NORWEGIAN_NUMERALS,
    SWEDISH_NUMERALS,
)


def generate_answer_candidates(answer: str, language: str) -> List[str]:
    """Generate answer candidates from an answer.

    Args:
        answer (str):
            The answer to generate candidates from.
        language (str):
            The language of the answer. Must be one of "en", "da", "sv" and "no".

    Returns:
        List[str]:
            The answer candidates.
    """
    # Create singleton list of answer candidates
    answer_candidates = [answer]

    # If the answer looks like an integer, then add the corresponding written form of
    # the integer to the answer candidates
    if re.match(r"^[0-9]+(\.0)?$", answer) is not None:

        # Extract the integer
        integer = int(re.sub(r"\.0", "", answer))

        # Get the list of numerals based on the language
        if language == "en":
            numerals = ENGLISH_NUMERALS
        elif language == "da":
            numerals = DANISH_NUMERALS
        elif language == "sv":
            numerals = SWEDISH_NUMERALS
        elif language == "no":
            numerals = NORWEGIAN_NUMERALS
        else:
            raise ValueError(f"Language {language} not supported.")

        # Add the written form of the integer to the answer candidates
        if integer >= 0 and integer <= 20:
            answer_candidates.extend(numerals[integer])

    return answer_candidates


def extract_answer(
    answer: Optional[str], context: str, language: str
) -> Union[dict, None]:
    """Extract the answer from the context.

    Args:
        answer (str or None):
            The answer from which to generate answer candidates. If None then None will
            always be outputted.
        context (str):
            The context where the answer candidates need to be located.
        language (str):
            The language of the context. Must be one of "en", "da", "sv" and "no".

    Returns:
        dict or None:
            The answer, with keys 'answer' and 'answer_start'. If no answer is found,
            returns None.

    Raises:
        ValueError:
            If the language is not supported.
    """
    # If the answer is None, then return None
    if answer is None:
        return None

    # Get list of answer candidates
    answer_candidates = generate_answer_candidates(answer=answer, language=language)

    # Create variable storing whether any of the answer candidates appear in the
    # translated context
    has_answer = any(
        candidate.lower() in context.lower() for candidate in answer_candidates
    )

    # If none of the answer candidates appear in the translated context then we return
    # None
    if not has_answer:
        return None

    # Otherwise, we extract the answer and answer index from the context
    else:

        # Extract the answer candidate appearing in the context
        answer = next(
            candidate
            for candidate in answer_candidates
            if candidate.lower() in context.lower()
        )

        # Get the index at which the answer appears in the context
        answer_idx = context.lower().index(answer.lower())

        # Use the index to extract the answer with correct casing from the context
        answer = context[answer_idx : answer_idx + len(answer)]

        # Return the answer and answer index
        return dict(answer=answer, answer_start=answer_idx)
