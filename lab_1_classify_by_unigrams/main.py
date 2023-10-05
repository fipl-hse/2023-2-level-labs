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
    tokens = [el.lower() for el in text if el.isalpha()]
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    frequency = {}
    quantity = len(tokens)
    for token in tokens:
        frequency.setdefault(token, 0)
        if not isinstance(token, str):
            return None
        if token in frequency:
            frequency[token] += 1
    for keys, values in frequency.items():
        frequency[keys] = values / quantity
    return frequency


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not (isinstance(language, str) and isinstance(text, str)):
        return None
    freq = calculate_frequencies(tokenize(text))
    return {'name': language,
            'freq': freq}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not len(predicted) == len(actual):
        return None
    if not (isinstance(predicted, list) and isinstance(actual, list)):
        return None
    total = 0
    for el1 in predicted:
        for el2 in actual:
            total += (el2 - el1) ** 2
    return total / len(actual)


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
    if not (isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict)):
        return None
    if ('name' and 'freq') not in unknown_profile and ('name' and 'freq') not in profile_to_compare:
        return None
    keys = sorted(set(unknown_profile.keys()) | set(profile_to_compare.keys()))
    compare_un = []
    compare_pr = []
    for key in keys:
        compare_un.append(unknown_profile.get(key, 0))
        compare_pr.append(profile_to_compare.get(key, 0))
    mse_calc = calculate_mse(compare_un, compare_pr)
    return mse_calc


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
    if not (
            isinstance(unknown_profile, dict) and
            isinstance(profile_1, dict) and
            isinstance(profile_2, dict)
    ):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if isinstance(mse_1, float) and isinstance(mse_2, float):
        if mse_1 == mse_2:
            names = [profile_1.get('name'), profile_2.get('name')]
            names = str(sorted(names))
            return names
        if mse_2 > mse_1:
            return str(profile_1.get('name'))
        if mse_1 > mse_2:
            return str(profile_2.get('name'))


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
