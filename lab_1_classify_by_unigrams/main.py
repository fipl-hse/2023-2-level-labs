"""
Lab 1
Language detection
"""


def tokenize(text):
    text = text.lower()
    for i in text:
        if not i.isalnum() and i != ' ':
            text = text.replace(i, '')
    for i in text:
        if i == ' ':
            text = text.replace(' ','')
    text = list(text)
    return text
tokens = tokenize(text)





def calculate_frequencies(tokens):
    dictionary = {}
    length = len(tokens)
    for i in tokens:
        dictionary[i] = (dictionary.setdefault(i, 0) + 1) / length




def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:


def calculate_mse(predicted: list, actual: list) -> float | None:



def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:



def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]],
) -> str | None:



def load_profile(path_to_file: str) -> dict | None:



def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:



def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:



def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:



def print_report(detections: list[list[str | float]]) -> None:

