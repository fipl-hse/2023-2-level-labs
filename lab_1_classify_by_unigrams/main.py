"""
Lab 1
Language detection
"""
import json


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    list1 = []

    if not isinstance(text, str):
        return None

    for i in text.lower():
        if i.isalpha():
            list1.append(i)

    return list1


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
        """
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None

    return dict((i, tokens.count(i) / len(tokens)) for i in tokens)


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    tokens = tokenize(text)
    frequency_dict = calculate_frequencies(tokens)

    if not frequency_dict:
        return None

    return {'name': language, 'freq': frequency_dict}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not (
            isinstance(predicted, list) and isinstance(actual, list)
            and len(predicted) == len(actual)
    ):
        return None

    difference = 0
    for i, pred in enumerate(predicted):
        difference += (pred - actual[i]) ** 2

    return difference / len(predicted)


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
    if not (
            isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict)
            and 'freq' in unknown_profile
            and 'freq' in profile_to_compare
            and 'name' in unknown_profile
            and 'name' in profile_to_compare
    ):
        return None

    unknown = []
    to_compare = []
    all_fregs = set(unknown_profile.get('freq').keys()) | set(profile_to_compare.get('freq').keys())
    for cur_tok in all_fregs:
        unknown.append(unknown_profile['freq'].get(cur_tok, 0))
        to_compare.append(profile_to_compare['freq'].get(cur_tok, 0))

    return calculate_mse(unknown, to_compare)


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
    if (
            not isinstance(unknown_profile, dict) or
            not isinstance(profile_1, dict) or
            not isinstance(profile_2, dict)
    ):
        return None

    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)

    if not (
            isinstance(mse1, float)
            and isinstance(mse2, float)
    ):
        return None

    if mse1 < mse2:
        return str(profile_1['name'])
    if mse1 > mse2:
        return str(profile_2['name'])
    return [profile_1['name'], profile_2['name']].sort()


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None

    with open(path_to_file, 'r', encoding='utf-8') as f:
        profile = json.load(f)

    if not isinstance(profile, dict):
        return None

    return profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not (
            isinstance(profile, dict)
            and "name" in profile
            and "freq" in profile
            and "n_words" in profile
    ):
        return None

    profile_new = {
        "name": profile["name"],
        "freq": {}
    }

    for i in profile["freq"]:
        if i.lower() in profile_new["freq"]:
            profile_new["freq"][i.lower()] += profile["freq"][i] / profile["n_words"][0]

        elif len(i) == 1 and i.isalpha:
            profile_new["freq"][i.lower()] = profile["freq"][i] / profile["n_words"][0]

    return profile_new


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not (
            isinstance(paths_to_profiles, list)
            and all(isinstance(path, str) for path in paths_to_profiles)
    ):
        return None

    profiles = []

    for path in paths_to_profiles:

        loaded_profile = load_profile(path)
        if load_profile(path) is None:
            return None
        preprocessed_profile = preprocess_profile(loaded_profile)
        if preprocess_profile(loaded_profile) is None:
            return None
        profiles.append(preprocessed_profile)
        if not preprocessed_profile:
            return None

    return profiles


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if not (
        isinstance(unknown_profile, dict)
        and isinstance(known_profiles, list)
    ):
        return None

    langs = [(i, compare_profiles(i, unknown_profile)) for i in known_profiles]

    langs = sorted(langs, key=lambda x: (x[1], x[0]))

    return langs


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if not isinstance(detections, list):
        for detect in detections:
            print(f'{detect[0]}: MSE {detect[1]:.5f}')
