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
    punc = ''''1234567890!"#$%&'()’*º+,-./:;<=>?@[\\]^_`{|}~'''
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    tokens = [token.lower() for token in text if token not in punc]
    return [*tokens]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    my_dict = {token: tokens.count(token) / len(tokens) for token in tokens}
    return my_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not (isinstance(language, str) and isinstance(text, str)):
        return None
    text_for_freq = tokenize(text)
    prof_dict = calculate_frequencies(text_for_freq)
    lang_profile = {"name": language, "freq": prof_dict}
    return lang_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not (isinstance(predicted, list)
            and isinstance(actual, list)
            and len(predicted) == len(actual)):
        return None
    mse_calculation = 0
    for num, act in enumerate(actual):
        mse_calculation += (act - predicted[num])**2
    return mse_calculation / len(actual)


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
    if not (isinstance(unknown_profile, dict)
            and isinstance(profile_to_compare, dict)
            and 'name' in unknown_profile
            and 'freq' in unknown_profile
            and 'name' in profile_to_compare
            and 'freq' in profile_to_compare):
        return None
    unknown_tokens = set(unknown_profile.get('freq').keys())
    compare_tokens = set(profile_to_compare.get('freq').keys())
    all_tokens = list(unknown_tokens.union(compare_tokens))
    list_actual = []
    list_predicted = []
    for token in all_tokens:
        if token not in unknown_profile["freq"]:
            list_actual.append(float(0))
            list_predicted.append(float(profile_to_compare['freq'][token]))
        elif token not in profile_to_compare['freq']:
            list_actual.append(float(unknown_profile["freq"][token]))
            list_predicted.append(float(0))
        else:
            list_actual.append(float(unknown_profile["freq"][token]))
            list_predicted.append(float(profile_to_compare['freq'][token]))
    return calculate_mse(list_predicted, list_actual)


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
    unknown_and_1 = compare_profiles(unknown_profile, profile_1)
    unknown_and_2 = compare_profiles(unknown_profile, profile_2)
    if unknown_and_1 < unknown_and_2:
        return profile_1['name']
    if unknown_and_2 < unknown_and_1:
        return profile_2['name']
    return sorted([profile_1['name'], profile_2['name']])[0]


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
