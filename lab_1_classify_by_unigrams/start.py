"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import create_language_profile, detect_language


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    en_lang = create_language_profile('en', en_text)
    de_lang = create_language_profile('de', de_text)
    un_lang = create_language_profile('un', unknown_text)
    if (
            isinstance(un_lang, dict) and
            isinstance(en_lang, dict) and
            isinstance(de_lang, dict)
    ):
        result = detect_language(un_lang, en_lang, de_lang)
        assert result, "Detection result is None"


if __name__ == "__main__":
    main()
