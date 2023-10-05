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
    if isinstance(text, str) is False:
        return None
    text = text.lower()
    return [i for i in text if i.isalpha()]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if (isinstance(tokens, list) is False
        or tokens == []
        or not all(isinstance(symbol, str) for symbol in tokens)):
        return None
    symbols = set(tokens)
    freq_dict = {}
    for letter in symbols:
        freq_dict[letter] = 0
    for symbol in tokens:
        if symbol in freq_dict:
            freq_dict[symbol] += 1 / len(tokens)
    return freq_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if isinstance(language, str) is False or isinstance(text, str) is False:
        return None
    return {"name": language, "freq": calculate_frequencies(tokenize(text))}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if (isinstance(predicted, list) is False or
            isinstance(actual, list) is False or
            len(predicted) != len(actual)):
        return None
    mse = 0
    for i, j in enumerate(predicted):
        mse += (predicted[i] - actual[i])**2 / len(predicted)
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
    if (isinstance(unknown_profile, dict) is False or
        isinstance(profile_to_compare, dict) is False or
        ("name" or "freq") not in unknown_profile or
        ("name" or "freq") not in profile_to_compare):
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
    return calculate_mse(list_p1, list_p2)


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
    if (isinstance(unknown_profile, dict) is False or
        isinstance(profile_1, dict) is False or
        isinstance(profile_2, dict) is False):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 < mse_2:
        return profile_1.get("name")
    if mse_1 == mse_2:
        alph = [profile_1.get("name"), profile_2.get("name")]
        alph = sorted(alph)
        return alph[0]
    if mse_1 > mse_2:
        return profile_2.get("name")

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
