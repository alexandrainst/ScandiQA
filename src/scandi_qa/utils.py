"""Utility functions and variables for the project."""

from typing import Dict, List


def get_numerals(language: str) -> Dict[int, List[str]]:
    """Get the mapping of numerals associated to a given language.

    Args:
        language (str):
            The language to get the numerals for.

    Returns:
        Dict[int, List[str]]:
            The mapping of numerals associated to the language.
    """
    if language == "en":
        return ENGLISH_NUMERALS
    elif language == "da":
        return DANISH_NUMERALS
    elif language == "sv":
        return SWEDISH_NUMERALS
    elif language == "no":
        return NORWEGIAN_NUMERALS
    else:
        raise ValueError(f"Unknown language: {language}")


def get_months(language: str) -> Dict[int, str]:
    """Get the mapping of months associated to a given language.

    Args:
        language (str):
            The language to get the months for.

    Returns:
        Dict[int, str]:
            The mapping of months associated to the language.
    """
    if language == "en":
        return ENGLISH_MONTHS
    elif language == "da":
        return DANISH_MONTHS
    elif language == "sv":
        return SWEDISH_MONTHS
    elif language == "no":
        return NORWEGIAN_MONTHS
    else:
        raise ValueError(f"Unknown language: {language}")


ENGLISH_NUMERALS = {
    1: ["one", "first"],
    2: ["two", "second"],
    3: ["three", "third"],
    4: ["four", "fourth"],
    5: ["five", "fifth"],
    6: ["six", "sixth"],
    7: ["seven", "seventh"],
    8: ["eight", "eighth"],
    9: ["nine", "ninth"],
    10: ["ten", "tenth"],
    11: ["eleven", "eleventh"],
    12: ["twelve", "twelfth"],
    13: ["thirteen", "thirteenth"],
    14: ["fourteen", "fourteenth"],
    15: ["fifteen", "fifteenth"],
    16: ["sixteen", "sixteenth"],
    17: ["seventeen", "seventeenth"],
    18: ["eighteen", "eighteenth"],
    19: ["nineteen", "nineteenth"],
    20: ["twenty", "twentieth"],
}


DANISH_NUMERALS = {
    0: ["nul"],
    1: ["en", "et", "første"],
    2: ["to", "andet", "anden"],
    3: ["tre", "tredje"],
    4: ["fire", "fjerde"],
    5: ["fem", "femte"],
    6: ["seks", "sjette"],
    7: ["syv", "syvende"],
    8: ["otte", "ottende"],
    9: ["ni", "niende"],
    10: ["ti", "tiende"],
    11: ["elleve", "ellevte"],
    12: ["tolv", "tolvte"],
    13: ["tretten", "trettende"],
    14: ["fjorten", "fjortende"],
    15: ["femten", "femtende"],
    16: ["seksten", "sekstende"],
    17: ["sytten", "syttende"],
    18: ["atten", "attende"],
    19: ["nitten", "nittende"],
    20: ["tyve", "tyvende"],
}


SWEDISH_NUMERALS = {
    0: ["noll"],
    1: ["en", "ett", "första"],
    2: ["två", "andra"],
    3: ["tre", "tredje"],
    4: ["fyra", "fjärde"],
    5: ["fem", "femte"],
    6: ["sex", "sjätte"],
    7: ["sju", "sjunde"],
    8: ["åtta", "åttonde"],
    9: ["nio", "nionde"],
    10: ["tio", "tionde"],
    11: ["elva", "elfte"],
    12: ["tolv", "tolfte"],
    13: ["tretton", "trettonde"],
    14: ["fjorton", "fjortonde"],
    15: ["femton", "femtonde"],
    16: ["sexton", "sextonde"],
    17: ["sjutton", "sjuttonde"],
    18: ["arton", "artonde"],
    19: ["nitton", "nittonde"],
    20: ["tjugo", "tjugonde"],
}


NORWEGIAN_NUMERALS = {
    0: ["null"],
    1: ["én", "éi", "ett", "éin", "eitt", "første"],
    2: ["to", "annen", "andre"],
    3: ["tre", "tredje"],
    4: ["fire", "fjerde"],
    5: ["fem", "femte"],
    6: ["seks", "sjette"],
    7: ["sju", "syv", "sjuende"],
    8: ["åtte", "åttende"],
    9: ["ni", "niende"],
    10: ["ti", "tiende"],
    11: ["elleve", "ellevte"],
    12: ["tolv", "tolvte"],
    13: ["tretten", "trettende"],
    14: ["fjorten", "fjortende"],
    15: ["femten", "femtende"],
    16: ["seksten", "sekstende"],
    17: ["sytten", "syttende"],
    18: ["atten", "attende"],
    19: ["nitten", "nittende"],
    20: ["tjue", "tjuende"],
}


ENGLISH_MONTHS = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}


DANISH_MONTHS = {
    1: "januar",
    2: "februar",
    3: "marts",
    4: "april",
    5: "maj",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}


SWEDISH_MONTHS = {
    1: "januari",
    2: "februari",
    3: "mars",
    4: "april",
    5: "maj",
    6: "juni",
    7: "juli",
    8: "augusti",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}


NORWEGIAN_MONTHS = {
    1: "januar",
    2: "februar",
    3: "mars",
    4: "april",
    5: "mai",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "desember",
}


MKQA_LANGUAGES = {
    "ar",  # Arabic
    "da",  # Danish
    "de",  # German
    "en",  # English
    "es",  # Spanish
    "fi",  # Finnish
    "fr",  # French
    "he",  # Hebrew
    "hu",  # Hungarian
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "km",  # Khmer
    "ms",  # Malay
    "nl",  # Dutch
    "no",  # Norwegian
    "pl",  # Polish
    "pt",  # Portuguese
    "ru",  # Russian
    "sv",  # Swedish
    "th",  # Thai
    "tr",  # Turkish
    "vi",  # Vietnamese
    "zh",  # Chinese
    "zh_cn",  # Chinese (Simplified)
    "zh_hk",  # Chinese (Hong kong)
    "zh_tw",  # Chinese (Traditional)
}


DEEPL_LANGUAGES = {
    "bg",  # Bulgarian
    "zh",  # Chinese
    "zh_cn",  # Chinese (Simplified)
    "cz",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "de",  # German
    "el",  # Greek
    "hu",  # Hungarian
    "in",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "lv",  # Latvian
    "lt",  # Lithuanian
    "pl",  # Polish
    "pt",  # Portuguese
    "pt_br",  # Portuguese (Brazil)
    "ro",  # Romanian
    "ru",  # Russian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sv",  # Swedish
    "tu",  # Turkish
}
