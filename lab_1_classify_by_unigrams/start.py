"""
Language detection starter
"""


from lab_1_classify_by_unigrams.main import (create_language_profile, detect_language)
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

    unknown_prof = create_language_profile("unknown", unknown_text)
    en_prof = create_language_profile("en", en_text)
    de_prof = create_language_profile("de", de_text)
    if (isinstance(unknown_prof, dict)
            and isinstance(en_prof, dict)
            and isinstance(de_prof, dict)):
        result = detect_language(unknown_prof, en_prof, de_prof)
        assert result, "Detection result is None"


if __name__ == "__main__":
    main()
