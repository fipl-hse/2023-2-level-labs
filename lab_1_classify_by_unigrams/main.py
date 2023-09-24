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
    if not isinstance(text, str):
        return None
    tokens = []
    for symbol in text:
        if symbol.isalpha():
            tokens += symbol.lower()
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
        Calculates frequencies of given tokens
        :param tokens: a list of tokens
        :return: a dictionary with frequencies
        """
    if not isinstance(tokens, list) or not tokens or not all(isinstance(element, str) for element in tokens):
        return None
    frequencies = {}
    unit = 1 / len(tokens)
    for letter in tokens:
        if letter in frequencies:
            frequencies[letter] += unit
        else:
            frequencies[letter] = unit
    return frequencies


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
        Creates a language profile
        :param language: a language
        :param text: a text
        :return: a dictionary with two keys – name, freq
        """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    profile = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
    return profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
        Calculates mean squared error between predicted and actual values
        :param predicted: a list of predicted values
        :param actual: a list of actual values
        :return: the score
        """
    if not isinstance(predicted, list) or not isinstance(actual, list) or len(predicted) != len(actual):
        return None
    total = len(predicted)
    score = 0
    for n in range(total):
        score += (actual[n] - predicted[n]) ** 2 / total
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
    target_set = {'freq', 'name'}
    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_to_compare, dict)
            or len(target_set & set(unknown_profile) & set(profile_to_compare)) != len(target_set)):
        return None
    new_freq = {a: 0 for a in set(list(unknown_profile['freq']) + list(profile_to_compare['freq']))}
    freq_to_compare = new_freq.copy()
    new_freq.update(unknown_profile['freq'])
    freq_to_compare.update(profile_to_compare['freq'])
    distance = calculate_mse(list(new_freq.values()), list(freq_to_compare.values()))
    return distance


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
