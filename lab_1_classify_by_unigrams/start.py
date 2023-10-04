"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (create_language_profile,
                                             detect_language, print_report,
                                             detect_language_advanced, collect_profiles)


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
    result = None
    en_profile = create_language_profile('en', en_text)
    de_profile = create_language_profile('de', de_text)
    unk_profile = create_language_profile('unknown', unknown_text)
    result = detect_language(unk_profile, en_profile, de_profile)
    path_to_profiles = ['assets/profiles/de.json',
                        'assets/profiles/en.json',
                        'assets/profiles/es.json',
                        'assets/profiles/fr.json',
                        'assets/profiles/it.json',
                        'assets/profiles/ru.json',
                        'assets/profiles/tr.json']

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
