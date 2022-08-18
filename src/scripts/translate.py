"""Translate individual texts."""

from scandi_qa.translation import DeepLTranslator, GoogleTranslator


def main():
    google_translator = GoogleTranslator()
    deepl_translator = DeepLTranslator()
    while True:
        print()
        text = input("Enter text to translate: ")
        google_translation = google_translator.translate(text=text, target_lang="en")
        deepl_translation = deepl_translator.translate(text=text, target_lang="en")
        print(f"Google translation: {google_translation}")
        print(f"DeepL translation: {deepl_translation}")


if __name__ == "__main__":
    main()
