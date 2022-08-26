"""Extracting the answer from a context."""

import re
from typing import List, Optional, Union

from .translation import Translator
from .utils import (
    DANISH_NUMERALS,
    ENGLISH_NUMERALS,
    NORWEGIAN_NUMERALS,
    SWEDISH_NUMERALS,
    get_months,
    get_numerals,
)


def generate_answer_candidates(
    answer: str, answer_en: Optional[str], language: str, translator: Translator
) -> List[str]:
    """Generate answer candidates from an answer.

    Args:
        answer (str):
            The answer to generate candidates from.
        answer_en (str or None):
            The English answer from which to generate answer candidates. If None then
            no English answer candidates will be generated.
        language (str):
            The language of the answer. Must be one of "en", "da", "sv" and "no".
        translator (Translator):
            The translator used to translate the answer.

    Returns:
        List[str]:
            The answer candidates.
    """
    # Get the mapping of numerals based on the language
    numerals = get_numerals(language=language)

    # Get the mapping of months based on the language
    months = get_months(language=language)

    # Create singleton list of answer candidates
    answer_candidates = [answer]

    # Add the English answer to the list of answer candidates if it is not None
    if answer_en:
        answer_candidates.append(answer_en)

    # Add candidates where "the" is removed
    answer_candidates.append(re.sub(r"[Tt]he", "", answer).strip())

    # Add candidates where "." is removed, but only if it is not followed by a number
    answer_candidates.append(re.sub(r"\.(?![0-9])", "", answer).strip())

    # Add candidates where "," is removed, but only if it is not followed by a number
    answer_candidates.append(re.sub(r"\,(?![0-9])", "", answer).strip())

    # Add candidates that spells out a date
    date_regex = r"(\d{4})-(\d{2})-(\d{2})"
    date_result = re.search(date_regex, answer)
    if date_result:
        year = int(date_result.group(1))
        month = int(date_result.group(2))
        day = int(date_result.group(3))
        month_description = months[month]
        answer_candidates.append(f"{day} {month_description} {year}")
        answer_candidates.append(f"{day} {month_description}, {year}")
        answer_candidates.append(f"{month_description} {day} {year}")
        answer_candidates.append(f"{month_description} {day}, {year}")
        answer_candidates.append(str(year))

    # Add candidates where we replace "." by "," and vice versa
    if "." in answer:
        answer_candidates.append(answer.replace(".", ","))
    elif "," in answer:
        answer_candidates.append(answer.replace(",", "."))

    # Add candidates where we remove trivial decimal endings from numbers
    if re.search(r"[0-9]+\.0", answer):
        answer_candidates.append(re.sub(r"([0-9]+)\.0", r"\1", answer))

    # If the answer looks like an integer, then add the corresponding written form of
    # the integer to the answer candidates, as well as the integer with thousand
    # separators
    if re.match(r"^[0-9]+(\.0)?$", answer) is not None:

        # Extract the integer
        integer = int(re.sub(r"\.0", "", answer))

        # Add the written form of the integer to the answer candidates
        if integer in numerals:
            answer_candidates.extend(numerals[integer])

        # Add the written form of the integer with thousand separators to the answer
        # candidates
        if integer in numerals:
            separated = f"{integer:,}"
            for thousand_separator in [".", ",", " "]:
                separated = separated.replace(",", thousand_separator)
                answer_candidates.append(separated)

    # Add the translation of the answer to the desired language to the answer
    # candidates
    translated_answers = [
        translator.translate(text=cand, target_lang=lang)
        for cand in answer_candidates + [answer]
        for lang in {language, "en"}
    ]
    answer_candidates.extend(translated_answers)

    # Manually add numerals in the target language, if the answer is a descriptive
    # numeral
    numeral_mappings = [
        ENGLISH_NUMERALS,
        DANISH_NUMERALS,
        SWEDISH_NUMERALS,
        NORWEGIAN_NUMERALS,
    ]
    for numeral_mapping in numeral_mappings:
        for numeral, descriptors in numeral_mapping.items():
            if answer in descriptors:
                answer_candidates.append(str(numeral))
                answer_candidates.extend(descriptors)
                answer_candidates.extend(numerals[numeral])

    # Remove empty answer candidates and duplicated answer candidates
    answer_candidates = list(
        {
            candidate
            for candidate in answer_candidates
            if re.search(r"[a-zæøåA-ZÆØÅ0-9]", candidate)
        }
    )

    # Sort the answer candidates by length, with the longest ones first
    answer_candidates = sorted(answer_candidates, key=len, reverse=True)

    return answer_candidates


def extract_answer(
    answer: Optional[str],
    answer_en: Optional[str],
    context: str,
    language: str,
    translator: Translator,
) -> Union[dict, None]:
    """Extract the answer from the context.

    Args:
        answer (str or None):
            The answer from which to generate answer candidates. If None then None will
            always be outputted.
        answer_en (str or None):
            The English answer from which to generate answer candidates. If None then
            no English answer candidates will be generated.
        context (str):
            The context where the answer candidates need to be located.
        language (str):
            The language of the context. Must be one of "en", "da", "sv" and "no".
        translator (Translator):
            The translator used to translate the answer.

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
    answer_candidates = generate_answer_candidates(
        answer=answer, answer_en=answer_en, language=language, translator=translator
    )

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

        orig_answer = answer

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

        if orig_answer != answer:
            print()
            print("=============")
            print(f"Answer conversion: {orig_answer} --> {answer}")
            print(f"Answer candidates: {answer_candidates}")
            print(f"Context: {context}")
            print("=============")
            print()

        # Return the answer and answer index
        return dict(answer=answer, answer_start=answer_idx)
