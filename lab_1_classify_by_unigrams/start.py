"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language)
def main() -> None:
    """
    Launches an implementation
    """
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
        result = detect_language(unknown, known)
    if result:
        print_report(result)

    assert result, "Detection result is None"

    import main



if __name__ == "__main__":
    main()
