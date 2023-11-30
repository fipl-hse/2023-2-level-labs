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
    characters = []
    for char in text.lower():
        if char.isalpha():
            characters.append(char)
    return characters


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    if not all(isinstance(token, str) for token in tokens):
        return None

    frequency = {}
    for i in tokens:
        frequency[i] = tokens.count(i) / len(tokens)
    return frequency


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str):
        return None
    if not isinstance(text, str):
        return None

    tokenized = tokenize(text)
    frequencies = calculate_frequencies(tokenized)
    profile = dict({"name": language, "freq": frequencies})
    if profile:
        return profile
    return None


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list):
        return None
    if not isinstance(actual, list):
        return None
    if not len(predicted) == len(actual):
        return None

    sum_diff = 0
    for freq_value in zip(predicted, actual):
        sum_diff += (freq_value[0] - freq_value[1]) ** 2
    mse = sum_diff / len(predicted)
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
    if not isinstance(unknown_profile, dict):
        return None
    if 'name' not in unknown_profile or 'freq' not in unknown_profile:
        return None
    if not isinstance(profile_to_compare, dict):
        return None
    if 'name' not in profile_to_compare or 'freq' not in profile_to_compare:
        return None

    all_tokens = set(unknown_profile['freq'].keys()) | set(profile_to_compare['freq'].keys())
    lang1_freq = []
    lang2_freq = []
    for token in all_tokens:
        lang1_freq.append(unknown_profile['freq'].get(token, 0))
        lang2_freq.append(profile_to_compare['freq'].get(token, 0))

    return calculate_mse(lang1_freq, lang2_freq)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) \
            or not isinstance(profile_2, dict):
        return None

    profile_1_mse = compare_profiles(unknown_profile, profile_1)
    profile_2_mse = compare_profiles(unknown_profile, profile_2)

    if not isinstance(profile_1_mse, float) or not isinstance(profile_2_mse, float):
        return None

    if profile_1_mse > profile_2_mse:
        return str(profile_2['name'])
    if profile_1_mse < profile_2_mse:
        return str(profile_1['name'])

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
