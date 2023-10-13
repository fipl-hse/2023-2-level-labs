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
    tokens_list = []
    for i in text:
        if i.isalpha():
            tokens_list += i.lower()
    return tokens_list


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    freq_dict = {}
    for i in tokens:
        if not isinstance(i, str):
            return None
        if i in freq_dict:
            freq_dict[i] += 1
        else:
            freq_dict[i] = 1
    for i in freq_dict:
        freq_dict[i] = freq_dict[i] / len(tokens)
    return freq_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    calculated_freq = calculate_frequencies(tokenize(text))
    if not isinstance(calculated_freq, dict):
        return None
    created_profile = {"name": language, "freq": calculated_freq}
    return created_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    mse = 0
    for value1, value2 in zip(predicted, actual):
        mse += (value1 - value2) ** 2
    return mse / len(predicted)


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
    if (not ("name" or "freq") in unknown_profile.keys() or
            not ("name" or "freq") in profile_to_compare.keys()):
        return None
    freq_unknown_profile = unknown_profile["freq"]
    freq_profile_to_compare = profile_to_compare["freq"]
    union_profiles = list(freq_unknown_profile.keys())[:]
    for i in freq_profile_to_compare:
        if i not in union_profiles:
            union_profiles += i
    freq_list_unknown = []
    freq_list_compare = []
    for i in union_profiles:
        freq_list_unknown.append(freq_unknown_profile.get(i, 0.0))
        freq_list_compare.append(freq_profile_to_compare.get(i, 0.0))
    return calculate_mse(freq_list_unknown, freq_list_compare)


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
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    comp_profile_1 = compare_profiles(unknown_profile, profile_1)
    comp_profile_2 = compare_profiles(unknown_profile, profile_2)
    if (not isinstance(comp_profile_1, float)
            or not isinstance(comp_profile_2, float)):
        return None
    if comp_profile_1 < comp_profile_2:
        return str(profile_1["name"])
    if comp_profile_1 > comp_profile_2:
        return str(profile_2["name"])
    if comp_profile_1 == comp_profile_2:
        profile_both = [profile_1['name'], profile_2['name']]
        profile_both.sort()
        return str(profile_both[0])


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
