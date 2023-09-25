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
    tokens_list = []
    if type(text) != str:
        return None
    for i in text:
        if i.isalpha():
            tokens_list.append(i.lower())
    return tokens_list


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if type(tokens) != list or tokens == []:
        return None
    count = 0
    for symbol in tokens:
        if type(symbol) != str:
            count += 1
    if count != 0:
        return None
    freq_dict = {}
    for symbol in tokens:
        if symbol in freq_dict:
            freq_dict[symbol] += 1 / len(tokens)
        else:
            freq_dict[symbol] = 1 / len(tokens)
    return freq_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if type(language) != str or type(text) != str:
        return None
    language_profile = {"name": language,
                        "freq": calculate_frequencies(tokenize(text))}
    return language_profile




def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if (type(predicted) != list or
            type(actual) != list or
            len(predicted) != len(actual)):
        return None
    mse_sum = 0
    for i in range(len(predicted)):
        mse_sum += (predicted[i] - actual[i])**2
    mse = mse_sum / len(predicted)
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
    if (type(unknown_profile) != dict or
        type(profile_to_compare) != dict or
            ("name" or "freq") not in (unknown_profile or profile_to_compare)):
        return None
    list_p1 = []
    list_p2 = []
    dict_p1 = unknown_profile.get("freq")
    dict_p2 = profile_to_compare.get("freq")
    for symbol in dict_p1:
        if symbol in dict_p2:
            list_p1.append(dict_p1.get(symbol))
            list_p2.append(dict_p2.get(symbol))
        else:
            list_p1.append(dict_p1.get(symbol))
            list_p2.append(0)
    for symbol in dict_p2:
        if symbol not in dict_p1:
            list_p1.append(0)
            list_p2.append(dict_p2.get(symbol))
    calculate_mse(list_p1, list_p2)



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
