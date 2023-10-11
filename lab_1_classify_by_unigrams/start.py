"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (create_language_profile, detect_language,
                                             detect_language_advanced, load_profile,
                                             preprocess_profile, print_report)

from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language_advanced, print_report)


def main() -> None:
    """
    Launches an implementation
    """
    paths_to_files = ['assets/profiles/de.json',
                      'assets/profiles/en.json',
                      'assets/profiles/es.json',
                      'assets/profiles/fr.json',
                      'assets/profiles/it.json',
                      'assets/profiles/ru.json',
                      'assets/profiles/tr.json'
                     ]
    language_profiles = []
    for path in paths_to_files:
        language_profile = load_profile(path)
        if language_profile:
            processed_profile = preprocess_profile(language_profile)
            if processed_profile:
                language_profiles.append(processed_profile)

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    json_paths = ['assets/profiles/de.json',
                  'assets/profiles/en.json',
                  'assets/profiles/es.json',
                  'assets/profiles/fr.json',
                  'assets/profiles/it.json',
                  'assets/profiles/ru.json',
                  'assets/profiles/tr.json']

    unknown = create_language_profile('unknown', unknown_text)
    known = collect_profiles(json_paths)
    if isinstance(unknown, dict) and isinstance(known, list):
        result = detect_language_advanced(unknown, known)
    if result:
        print_report(result)

    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
