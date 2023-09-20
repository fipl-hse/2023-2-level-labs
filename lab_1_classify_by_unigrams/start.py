"""
Language detection starter
"""
from main import (
    tokenize,
    calculate_frequencies,
    create_language_profile,
    compare_profiles,
    detect_language,
    collect_profiles,
    detect_language_advanced,
    print_report
)


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
    assert result, "Detection result is None"
    en_tokens = tokenize(en_text)
    de_tokens = tokenize(de_text)
    unknown_tokens = tokenize(unknown_text)
    en_profile = create_language_profile(calculate_frequencies(en_tokens))
    de_profile = create_language_profile(calculate_frequencies(de_tokens))
    unknown_profile = create_language_profile(calculate_frequencies(unknown_tokens))
    detected_language = detect_language(unknown_profile, en_profile, de_profile)
    paths_to_profiles = [
        'assets/profiles/de.json',
        'assets/profiles/en.json',
        'assets/profiles/es.json',
        'assets/profiles/fr.json',
        'assets/profiles/it.json',
        'assets/profiles/ru.json',
        'assets/profiles/tr.json'
    ]
    preprocessed_profiles = collect_profiles(paths_to_profiles)
    frequencies = detect_language_advanced(unknown_profile, preprocessed_profiles)
    print_report(frequencies)


if __name__ == "__main__":
    main()


