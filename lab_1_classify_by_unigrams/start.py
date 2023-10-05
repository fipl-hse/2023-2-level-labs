"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (load_profile, preprocess_profile,
                                             create_language_profile,
                                             detect_language, detect_language_advanced,
                                             print_report)


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

    en_profile = create_language_profile('en', en_text)
    de_profile = create_language_profile('de', de_text)
    unknown_profile = create_language_profile('unknown', unknown_text)

    if (isinstance(en_profile, dict)
                  and isinstance(de_profile, dict)
                  and isinstance(unknown_profile, dict)):
        detect_language(unknown_profile, en_profile, de_profile)

    if isinstance(unknown_profile, dict) and isinstance(language_profiles, list):
        result = detect_language_advanced(unknown_profile, language_profiles)
        if result:
            print_report(result)

        assert result, "Detection result is None"

if __name__ == "__main__":
    main()
