"""
Language detection starter
"""
import lab_1_classify_by_unigrams.main


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        en_profile = lab_1_classify_by_unigrams.main.create_language_profile('en', en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_profile = lab_1_classify_by_unigrams.main.create_language_profile('de', de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = lab_1_classify_by_unigrams.main.create_language_profile('unknown', unknown_text)
    result = None
    assert result, "Detection result is None"
    result = lab_1_classify_by_unigrams.main.detect_language(unknown_profile, en_profile, de_profile)
    return result

if __name__ == "__main__":
    main()
