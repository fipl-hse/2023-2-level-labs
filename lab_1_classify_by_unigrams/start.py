"""
Language detection starter
"""
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
    eng_pr = create_language_profile('English', en_text)
    de_pr = create_language_profile('Deutsch', de_text)
    unk_pr = create_language_profile('Unknown', unknown_text)
    assert eng_pr
    assert de_pr
    assert unk_pr
    result = detect_language(unk_pr, eng_pr, de_pr)
    assert result
    print('Detection result is ' + str(result))


if __name__ == "__main__":
    main()
