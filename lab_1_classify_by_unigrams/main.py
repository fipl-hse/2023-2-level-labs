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
    for symbol in text:
        if symbol.isalpha():
            tokens.append(symbol)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list) or not all(isinstance(el, str) for el in tokens):
        return None
    freqs = {}
    for token in tokens:
        if token in freqs:
            freqs[token] += 1
        else:
            freqs[token] = 1
    for k, v in freqs.items():
        freqs[k] = v / len(tokens)
    return freqs


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    freqs = calculate_frequencies(tokenize(text))
    profile = {"name": language, "freq": freqs}
    return profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list) or (len(predicted) != len(actual)):
        return None
    dif = []
    for i in range(len(actual)):
        dif.append((actual[i] - predicted[i]) ** 2)
    mse = sum(dif) / len(dif)
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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict) \
            or ("name" not in unknown_profile.keys()) or ("freq" not in unknown_profile.keys()) \
            or ("name" not in profile_to_compare.keys()) or ("freq" not in profile_to_compare.keys()):
        return None
    all_symbols = set(unknown_profile.get("freq").keys()).union(set(profile_to_compare.get("freq").keys()))
    unknown_freq = []
    compare_freq = []
    for symbol in all_symbols:
        if symbol in unknown_profile.get("freq").keys():
            unknown_freq.append(unknown_profile.get("freq").get(symbol))
        else:
            unknown_freq.append(0)
        if symbol in profile_to_compare.get("freq").keys():
            compare_freq.append(profile_to_compare.get("freq").get(symbol))
        else:
            compare_freq.append(0)
    mse = calculate_mse(unknown_freq, compare_freq)
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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) \
            or not isinstance(profile_2, dict):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 == mse_2:
        profs_to_sort = [profile_1.get("name"), profile_2.get("name")]
        profs_to_sort.sort()
        return profs_to_sort[0]
    elif mse_1 < mse_2:
        return profile_1.get("name")
    else:
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
