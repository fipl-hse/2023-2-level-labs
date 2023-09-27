"""
Language detection starter
"""
from main import tokenize
from main import create_language_profile
from main import detect_language


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
    en_tokens = tokenize(en_text)
    print(en_tokens)
    print(create_language_profile('English', en_text))
    result = detect_language(create_language_profile('Unknown', unknown_text),
                             create_language_profile('English', en_text),
                             create_language_profile('Deutsch', de_text))
    print(f'Detection result is {result}')


if __name__ == "__main__":
    main()
