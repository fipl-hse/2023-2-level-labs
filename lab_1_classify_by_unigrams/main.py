"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str] or None:
    if isinstance(text, str):
        tokens = []
        text = text.lower()
        for symbol in text:
            if symbol.isalpha():
                tokens.append(symbol)
        return tokens
    else:
        return None


def calculate_frequencies(tokens: list[str] or None) -> dict[str, float] or None:
    if isinstance(tokens, list) and all(isinstance(el, str) for el in tokens):
        freqs = {}
        for token in tokens:
            if token in freqs:
                freqs[token] += 1
            else:
                freqs[token] = 1
        for k, v in freqs.items():
            freqs[k] = v / len(tokens)
        return freqs
    else:
        return None


def create_language_profile(language: str, text: str) -> dict[str, str or dict[str, float]] or None:
    if isinstance(language, str) and isinstance(text, str):
        freqs = calculate_frequencies(tokenize(text))
        profile = {"name": language, "freq": freqs}
        return profile
    else:
        return None


def calculate_mse(predicted: list, actual: list) -> float or None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """


def compare_profiles(
        unknown_profile: dict[str, str or dict[str, float]],
        profile_to_compare: dict[str, str or dict[str, float]]
) -> float or None:
    """
    Compares profiles and calculates the distance using symbols
    :param unknown_profile: a dictionary of an unknown profile
    :param profile_to_compare: a dictionary of a profile to compare the unknown profile to
    :return: the distance between the profiles
    """


def detect_language(
        unknown_profile: dict[str, str or dict[str, float]],
        profile_1: dict[str, str or dict[str, float]],
        profile_2: dict[str, str or dict[str, float]],
) -> str or None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """


def load_profile(path_to_file: str) -> dict or None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """


def preprocess_profile(profile: dict) -> dict[str, str or dict] or None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str or dict[str, float]]] or None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """


def detect_language_advanced(unknown_profile: dict[str, str or dict[str, float]],
                             known_profiles: list) -> list or None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
