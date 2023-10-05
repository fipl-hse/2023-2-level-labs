"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if isinstance(text, str):
        text = text.lower()
        tokens = []
        for symbol in text:
            if symbol.isalpha():
                tokens.append(symbol)
        return tokens
    Esle:
    return None


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if isinstance(tokens, list):    
        frequencies = {}
        for symbol in tokens:
            if symbol not in frequencies:
                frequencies[symbol] = 0
            frequencies[symbol] += 1
        for symbol, freq in frequencies:
            frequencies[symbol] = freq / len(tokens)
        return frequencies
    else:
        return None


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if isinstance(language, str) and isinstance(text, str):
        language_profile = {"name": language, "freq": calculate_frequencies(tokenize(text))}
        return language_profile
    else:
        return None


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if isinstance(predicted, list) and isinstance(actual, list):    
        number = 0
        total = 0
        for i, value in enumerate(actual):
            total += (value - predicted[i])**2
            number += 1
        score = total/number, 3
        return score
        return score
    else:
        return None


def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    """
    Compares profiles and calculates the distance using symbols
    :param unknown_profile: a dictionary of an unknown profile
    :param profile_to_compare: a dictionary of a profile to compare the unknown profile to
    :return: the distance between the profiles
    """
    if isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict) and 'name' in unknown_profile and 'name' in profile_to_compare:
        words = set(profile_to_compare["freq"].keys())
        words.update(unknown_profile["freq"].keys())
        listed_unknown_profile = []
        listed_profile_to_compare = []
        for letter in words:
            listed_profile_to_compare.append(profile_to_compare["freq"].get(letter, 0))
            listed_unknown_profile.append(unknown_profile["freq"].get(letter, 0))
        profile_distance = calculate_mse(listed_profile_to_compare, listed_unknown_profile)
        return profile_distance
    else:
        return None


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """
    if isinstance(unknown_profile, dict) and isinstance(profile_1, dict) and isinstance(profile_2, dict):
        profile_distance_1 = compare_profiles(unknown_profile, profile_1)
        profile_distance_2 = compare_profiles(unknown_profile, profile_2)
        if profile_distance_1 < profile_distance_2:
            return profile_1["name"]
        elif profile_distance_2 < profile_distance_1:
            return profile_2["name"]
        elif profile_distance_1 == profile_distance_2:
            language_list = sorted([str(profile_1["name"]), str(profile_2["name"])])
            return language_list[0]
    else:
        return None


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
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
