from lab_1_classify_by_unigrams.main import create_language_profile, detect_language


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
    result = detect_language(create_language_profile('Unknown', unknown_text),
                             create_language_profile('English', en_text),
                             create_language_profile('Deutsch', de_text))
    assert result, "Detection result is None"
    print(f'Detection result is {result}')


if __name__ == "__main__":
    main()
