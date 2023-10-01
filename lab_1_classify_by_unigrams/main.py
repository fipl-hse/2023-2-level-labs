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

    split_text = list(text.lower())
    return [i for i in split_text if i.isalpha()]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None

    for i in tokens:
        if not isinstance(i, str):
            return None
    return {i: tokens.count(i) / len(tokens) for i in tokens}


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """

    if not (isinstance(language, str) and isinstance(text, str)):
        return None

    tokens = tokenize(text)
    freq_dict = {
        "name": language,
        "freq": calculate_frequencies(tokens)
    }
    if not isinstance(freq_dict, dict):
        return None

    return freq_dict


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if (
            not (isinstance(predicted, list)
                 and isinstance(actual, list))
            or len(predicted) != len(actual)
    ):
        return None
    difference = [(i - j) ** 2 for i, j in zip(actual, predicted)]
    return sum(difference) / len(actual)


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
    if (
            not (isinstance(unknown_profile, dict)
                 or isinstance(profile_to_compare, dict))
            or ('name' or 'freq') not in unknown_profile
            or ('name' or 'freq') not in profile_to_compare
    ):
        return None

    all_tokens = set(unknown_profile['freq'].keys()) | set(profile_to_compare['freq'].keys())
    language1 = [unknown_profile['freq'].get(token, 0) for token in all_tokens]
    language2 = [profile_to_compare['freq'].get(token, 0) for token in all_tokens]
    return calculate_mse(language1, language2)


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
    if not (isinstance(unknown_profile, dict)
            and isinstance(profile_1, dict)
            and isinstance(profile_2, dict)):
        return None

    distance1 = compare_profiles(unknown_profile, profile_1)
    distance2 = compare_profiles(unknown_profile, profile_2)

    if distance1 > distance2:
        language_detected = str(profile_2['name'])
    if distance1 < distance2:
        language_detected = str(profile_1['name'])
    if distance1 == distance2:
        language_detected = [profile_1['name'], profile_2['name']].sort()
    return language_detected


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
