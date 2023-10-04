"""
Language detection starter
"""

from lab_1_classify_by_unigrams.main import tokenize, calculate_frequencies, create_language_profile


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

    tokens = tokenize(en_text)
    print(tokens)
    print(calculate_frequencies(tokens))
    print(create_language_profile('en', en_text))

    #result = None
    #assert result, "Detection result is None"
    #en_tokens = lab_1_classify_by_unigrams.main.tokenize(en_text)
    #print(en_tokens)
    #language_profile = lab_1_classify_by_unigrams.main.create_language_profile('en', en_text)
    #print(language_profile)

if __name__ == "__main__":
    main()
