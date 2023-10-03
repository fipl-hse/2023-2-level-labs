"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import create_language_profile
from lab_1_classify_by_unigrams.main import detect_language
from lab_1_classify_by_unigrams.main import tokenize

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
    eng_pr = create_language_profile('English', en_text)
    print(eng_pr)
    unk_pr = create_language_profile('Unknown', unknown_text)
    de_pr = create_language_profile('Deutsch', de_text)
    result = detect_language(unk_pr, eng_pr, de_pr)
    print('Detection result is ' + str(result))


if __name__ == "__main__":
    main()
