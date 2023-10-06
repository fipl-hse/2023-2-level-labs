"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language_advanced, print_report)


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

    paths_to_profiles = ["assets/profiles/es.json",
                         "assets/profiles/de.json",
                         "assets/profiles/en.json",
                         "assets/profiles/fr.json",
                         "assets/profiles/it.json",
                         "assets/profiles/ru.json",
                         "assets/profiles/tr.json"]

    known_profiles = collect_profiles(paths_to_profiles)
    unknown_profile = create_language_profile("unknown", unknown_text)
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None
    result = detect_language_advanced(unknown_profile, known_profiles)
    if isinstance(result, list):
        print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
