"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import calculate_frequencies, create_language_profile, tokenize


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
    en_tokenized = tokenize(en_text)
    print(en_tokenized)
    en_freqs = calculate_frequencies(en_tokenized)
    en_profile = create_language_profile("eng", en_text)
    print(en_profile)
    result = en_profile
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
