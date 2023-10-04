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
    text = text.lower()
    tokens = []

    for word in text:
        for letter in word:
            if letter.isalpha():
                tokens.append(letter)

    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token,str):
            return None

    dict_of_freq = {}
    for token in tokens:
        quantity_of_token = tokens.count(token)
        frequency = quantity_of_token / len(tokens)
        dict_of_freq.update({token: frequency})

    return dict_of_freq


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language,str) or not isinstance(text,str):
        return None

    dict_of_freq = calculate_frequencies(tokenize(text))

    if isinstance(dict_of_freq, dict):
        return {'name': language, 'freq': dict_of_freq}
    return None

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

    mse = 0.0
    for index, actual_value  in enumerate(actual):
        mse += (actual_value - predicted[index])**2

    mse /= len(actual)
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
    if ('name' or 'freq') not in (unknown_profile or profile_to_compare):
        return None

    joint_elements = [[],[]]
    unknown_frequence = unknown_profile['freq']
    compare_frequence = profile_to_compare['freq']

    for symbol in compare_frequence:
        if symbol in unknown_frequence:
            joint_elements[0].append(unknown_frequence.get(symbol))
            joint_elements[1].append(compare_frequence.get(symbol))
        else:
            joint_elements[0].append(0.0)
            joint_elements[1].append(compare_frequence.get(symbol))

    for element in unknown_frequence:
        if element not in compare_frequence:
            joint_elements[1].append(0.0)
            joint_elements[0].append(unknown_frequence.get(element))

    return calculate_mse(joint_elements[0], joint_elements[1])

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

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)

    name_1 = str(profile_1['name'])
    name_2 = str(profile_2['name'])

    if isinstance(mse_1, float) and isinstance(mse_2, float):
        if mse_1 > mse_2:
            return name_2
        if mse_2 > mse_1:
            return name_1
        list_of_names = [name_1, name_2]
        list_of_names.sort()
        return list_of_names[0]
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
