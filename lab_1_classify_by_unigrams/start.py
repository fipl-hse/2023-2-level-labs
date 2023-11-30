"""
Language detection starter
"""

from lab_1_classify_by_unigrams.main import tokenize, create_language_profile, detect_language


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

    #mark 4
    en_tokens = tokenize(en_text)
    print(en_tokens)

    # mark 6
    en_profile = create_language_profile('en', en_text)
    print(en_profile)

    # mark 8
    de_profile = create_language_profile('de', de_text)
    unknown_profile = create_language_profile('unk', unknown_text)
    if (
            isinstance(unknown_profile, dict) and
            isinstance(en_profile, dict) and
            isinstance(de_profile, dict)
    ):
        result = detect_language(unknown_profile, en_profile, de_profile)
        print(result)
        assert result, "Detection result is None"


if __name__ == "__main__":
    main()
