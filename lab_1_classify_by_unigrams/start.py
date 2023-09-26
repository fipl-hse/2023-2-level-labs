"""
Language detection starter
"""

from main import (
    tokenize,
    calculate_frequencies,
    create_language_profile,
    calculate_mse,
    compare_profiles
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
    distance = compare_profiles(unknown_profile, en_profile, de_profile)



if __name__ == "__main__":
    main()



