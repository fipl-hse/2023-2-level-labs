"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language, detect_language_advanced,
                                             print_report)


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

    en_profile = create_language_profile('en', en_text)
    de_profile = create_language_profile('de', de_text)
    unknown_profile = create_language_profile('unknown', unknown_text)
    if (
        isinstance(unknown_profile, dict) and isinstance(en_profile, dict)
        and isinstance(de_profile, dict)
    ):
        detect_language(unknown_profile, en_profile, de_profile)

    path_to_profiles = [
        "assets/profiles/de.json",
        "assets/profiles/en.json",
        "assets/profiles/es.json",
        "assets/profiles/fr.json",
        "assets/profiles/it.json",
        "assets/profiles/ru.json",
        "assets/profiles/tr.json",
    ]

    preprocessed_profiles = collect_profiles(path_to_profiles)
    result = detect_language_advanced(unknown_profile, preprocessed_profiles)
    print_report(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
