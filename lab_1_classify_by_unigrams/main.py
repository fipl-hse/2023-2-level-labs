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
    text = text.lower()
    for i in text:
        if i.isalpha():
            tokens += i
    return tokens

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
    dictionary = {}
    length = len(tokens)
    for i in tokens:
        if i in dictionary:
            dictionary[i] += 1/length
        else:
            dictionary[i] = 1/length
    return dictionary

def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    new_dictionary = {}
    text = tokenize(text)
    freq = calculate_frequencies(text)
    new_dictionary['name'] = language
    new_dictionary['freq'] = freq
    return new_dictionary

def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or\
            not isinstance(actual, list) != list or\
            len(predicted) != len(actual):
        return None
    cubs = 0
    length = len(actual)
    for i, let in enumerate(actual):
        cubs += (actual[i] - predicted[i]) ** 2
    mse = cubs / length
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
    key1 = 'name'
    key2 = 'freq'
    if key1 not in unknown_profile or\
            key2 not in unknown_profile or\
            key1 not in profile_to_compare or\
            key2 not in profile_to_compare:
        return None
    unknown_list = []
    compare_list = []
    unknown_freq = unknown_profile['freq']
    compare_freq = profile_to_compare['freq']
    unknown = list(unknown_freq.keys())
    compare = list(compare_freq.keys())
    all_freq = list(set(unknown + compare))
    for i in all_freq:
        unknown_list.append(unknown_freq.get(i, 0))
        compare_list.append(compare_freq.get(i, 0))
    mse = calculate_mse(unknown_list, compare_list)
    return mse



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
    if not isinstance(unknown_profile, dict) or\
            not isinstance(profile_1, dict) or\
            not isinstance(profile_2, dict):
        return None
    if compare_profiles(unknown_profile, profile_1) < compare_profiles(unknown_profile, profile_2):
        return profile_1['name']
    if compare_profiles(unknown_profile, profile_1) == compare_profiles(unknown_profile, profile_2):
        names = list(profile_1['name'] + profile_2['name'])
        sorted_names = sorted(names)
        return sorted_names[0]
    return profile_2['name']


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
