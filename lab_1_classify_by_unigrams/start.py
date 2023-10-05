"""
Language detection starter
"""
import lab_1_classify_by_unigrams.main as main_py


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        en_profile = main_py.create_language_profile('en', en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_profile = main_py.create_language_profile('de', de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = main_py.create_language_profile('unknown', unknown_text)
    result = main_py.detect_language(unknown_profile, en_profile, de_profile)
    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
