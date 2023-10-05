"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (collect_profiles,
                                             create_language_profile,
                                            detect_language_advanced,
                                            print_report)


def main() -> None:
    """
    Launches an implementation
    """
    paths_to_profiles = ['assets/profiles/en.json', 'assets/profiles/es.json',
                         'assets/profiles/fr.json','assets/profiles/it.json',
                         'assets/profiles/ru.json','assets/profiles/tr.json']
    
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    unknown_profile = create_language_profile('unknown language', unknown_text)
    known_profiles = collect_profiles(paths_to_profiles)
    if isinstance(unknown_profile, dict) and \
        all(isinstance(profile, dict) for profile in known_profiles) \
        and isinstance(known_profiles, list):
        result = detect_language_advanced(unknown_profile, known_profiles)
        if isinstance(result, list):
            print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
