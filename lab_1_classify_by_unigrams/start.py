from lab_1_classify_by_unigrams.main import create_language_profile
from lab_1_classify_by_unigrams.main import detect_language_advanced
from lab_1_classify_by_unigrams.main import print_report
from lab_1_classify_by_unigrams.main import preprocess_profile
from lab_1_classify_by_unigrams.main import load_profile

"""
Language detection starter
"""


def main() -> None:
    """
    Launches an implementation
    """
    profiles_paths = ['assets/profiles/en.json',
                      'assets/profiles/es.json',
                      'assets/profiles/fr.json',
                      'assets/profiles/it.json',
                      'assets/profiles/ru.json',
                      'assets/profiles/tr.json',
                      ]
    language_profiles = []
    for path in profiles_paths:
        profile = load_profile(path)
        preprocessed_profile = preprocess_profile(profile)
        language_profiles.append(preprocessed_profile)

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    en_profile = create_language_profile('en', en_text)
    de_profile = create_language_profile('de', de_text)
    unk_profile = create_language_profile('unk', unknown_text)

    result = detect_language_advanced(unk_profile, [en_profile, de_profile])
    print('The distances to English and German languages:')
    print_report(result)

    result_complex = detect_language_advanced(unk_profile, language_profiles)
    print('The distances to En, Fr, It, Ru, Es and Tr languages:')
    print_report(result_complex)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
