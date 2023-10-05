"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language_advanced, print_report)


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    langs = ["de", "en", "es", "fr", "it", "ru", "tr"]
    paths = [f"assets/profiles/{langs[i]}.json" for i in range(7)]
    profiles = collect_profiles(paths)
    unknown_profile = create_language_profile("unknown", unknown_text)
    if not isinstance(profiles, list) or not isinstance(unknown_profile, dict):
        return
    result = detect_language_advanced(unknown_profile, profiles)
    if not result:
        return
    print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
