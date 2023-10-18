"""
Language detection starter
"""

from lab_1_classify_by_unigrams.main import create_language_profile, detect_language


def main() -> None:
    """
    Launches an implementation
    """

    en_text = file_to_read_en.read()
    de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    unkmown_p = create_language_profile("un", unknown_text)
    eng_p = create_language_profile("en", en_text)
    deu_p = create_language_profile("de", de_text)
    if (isinstance(unkmown_p, dict) and
            isinstance(eng_p, dict) and
            isinstance(deu_p, dict)):
        result = detect_language(unk, eng, deu)
        assert result, "Detection result is None"






if __name__ == "__main__":
    main()
