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
    unknown_profile = create_language_profile('Unknown', unknown_text)
    en_profile = create_language_profile('English', en_text)
    de_profile = create_language_profile('Deutsch', de_text)
    if not unknown_profile or not en_profile or not de_profile:
        return None
    result = detect_language(unknown_profile, en_profile, de_profile)
    if not result:
        return None
    print(f'Detection result is {result}')
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
