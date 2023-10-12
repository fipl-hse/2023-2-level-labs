"""
Language detection starter
"""


from main import tokenize
from main import calculate_frequencies
from main import create_language_profile
from main import calculate_mse
from main import compare_profiles
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
    assert result, "Detection result is None"
    tokenize(en_text)
    calculate_frequencies(tokenize(en_text))
    create_language_profile('en', en_text)
    calculate_mse()
    compare_profiles()

if __name__ == "__main__":
    main()
