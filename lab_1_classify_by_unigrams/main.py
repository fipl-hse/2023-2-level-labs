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
    try:
        text = text.lower()
    except TypeError:
        print("Incorrect input.")
    tokens = []
    for symbol in text:
        if symbol.isalpha():
            tokens.append(symbol)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    frequencies = {}
    for symbol in tokens:
        if symbol not in frequencies:
            frequencies[symbol] = 1
        else:
            frequencies[symbol] += 1
    for symbol, freq in frequencies:
        frequencies[symbol] = freq / len(tokens)
    return frequencies


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    language_profile = {"name": language, "freq": calculate_frequencies(tokenize(text))}
    return language_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    number = 0
    total = 0
    for i, value in enumerate(actual):
        total += (value - predicted[i])**2
        number += 1
    score = round(total/number, 3)
    return score


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
    values_unk = unknown_profile["freq"]
    values_comp = profile_to_compare["freq"]
    for k in values_unk:
        if k not in values_comp:
            values_comp[k] = 0
    for k in values_comp:
        if k not in values_unk:
            values_unk[k] = 0
    sorted_values_unk = dict(sorted(values_unk.items()))
    sorted_values_comp = dict(sorted(values_comp.items()))
    values_list_unk = list(sorted_values_unk.values())
    values_list_comp = list(sorted_values_comp.values())
    profile_distance = calculate_mse(values_list_comp, values_list_unk)
    return profile_distance


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
    profile_distance_1 = compare_profiles(unknown_profile, profile_1)
    profile_distance_2 = compare_profiles(unknown_profile, profile_2)
    if profile_distance_1 < profile_distance_2:
        return profile_1["name"]
    elif profile_distance_2 < profile_distance_1:
        return profile_2["name"]
    elif profile_distance_1 == profile_distance_2:
        language_list = []
        language_list.append(profile_1["name"])
        language_list.append(profile_2["name"])
        language_list = language_list.sort()
        return language_list[0]


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


def print_report(detections: list[list[str | float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
