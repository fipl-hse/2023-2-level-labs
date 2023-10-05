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
    new_text = []
    for symbol in text.lower():
        if symbol.isalpha():
            new_text += symbol
    return new_text


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """

    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None
    calc = {}
    for token in tokens:
        calc[token] = tokens.count(token) / len(tokens)
    return calc


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """

    if not isinstance(language, str) or not isinstance(text, str):
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    if not isinstance(freq_dict, dict):
        return None
    return {"name": language, "freq": freq_dict}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """

    if not (isinstance(predicted, list)
            and isinstance(actual, list)
            and len(predicted) == len(actual)
    ):
        return None
    difference_square = []
    quantity_of_meanings = len(predicted)
    for i in range(quantity_of_meanings):
        difference_square.append((actual[i] - predicted[i]) ** 2)
    mse = sum(difference_square)/quantity_of_meanings
    return mse


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

    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if not ('name' or 'freq') in unknown_profile or not ('name' or 'freq') in profile_to_compare:
        return None
    unknown = set(unknown_profile['freq'])
    comparable = set(profile_to_compare['freq'])
    set_of_tokens = unknown.union(comparable)
    unknown_freq = []
    to_compare_freq = []
    for token in set_of_tokens:
        unknown_freq.append(unknown_profile['freq'].get(token, 0))
        to_compare_freq.append(profile_to_compare['freq'].get(token, 0))
    mse_to_compare = calculate_mse(unknown_freq, to_compare_freq)
    return mse_to_compare


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]]
) -> str | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """

    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if isinstance(mse_1, float) and isinstance(mse_2, float):
        if mse_1 < mse_2:
            return str(profile_1['name'])
        if mse_2 < mse_1:
            return str(profile_2['name'])

    return [profile_1['name'], profile_2['name']].sort()


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
