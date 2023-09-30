"""
Language detection starter
"""
from main import create_language_profile
from main import detect_language

def main():
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    #result = None
    #assert result, "Detection result is None"
    en_text_profile = create_language_profile('en', en_text)
    de_text_profile = create_language_profile('de', de_text)
    unknown_text_profile = create_language_profile('unknown', unknown_text)
    print(detect_language(unknown_text_profile, en_text_profile, de_text_profile))





if __name__ == "__main__":
    main()
