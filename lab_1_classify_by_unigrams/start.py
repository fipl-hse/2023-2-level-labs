"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import tokenize, calculate_frequencies, create_language_profile, detect_language
import json


def main() -> None:
    """
    Launches an implementation
    """
    langs = ["de", "en", "es", "fr", "it", "ru", "tr"]
    paths = [f"assets/profiles/{langs[i]}.json" for i in range(7)]
    result = detect_language(unk, en, de)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
