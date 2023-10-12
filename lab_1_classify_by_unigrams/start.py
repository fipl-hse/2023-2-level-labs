"""
Language detection starter
"""


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
    unknownt = create_language_profile("unknown", unknown_text)
    givente = create_language_profile("en", en_text)
    giventd = create_language_profile("de", de_text)
    answer = detect_language(unknownt, givente, giventd)
    print(answer)

    assert result, "Detection result is None"

    import main



if __name__ == "__main__":
    main()
