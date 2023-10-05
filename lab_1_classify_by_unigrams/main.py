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
    if not isinstance(text, str):
        return None
    return [symbol.lower() for symbol in text if symbol.isalpha()]


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
    freqs = {}
    for token in tokens:
        if token in freqs:
            freqs[token] += 1
        else:
            freqs[token] = 1
    for token, freq in freqs.items():
        freqs[token] = freq / len(tokens)
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
    values_freq = calculate_frequencies(tokenize(text))
    if not isinstance(values_freq, dict):
        return None
    return {'name': language, 'freq': values_freq}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if (not isinstance(actual, list) or not isinstance(predicted, list) or
            len(actual) != len(predicted)):
        return None
    summ_values = 0
    squared_difference = ([(actual_value - predicted_value)**2
                           for actual_value, predicted_value in zip(actual, predicted)])
    for value in squared_difference:
        summ_values += value
    mse = round(summ_values / len(actual), 4)
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
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict) or
            'name' not in unknown_profile or 'name' not in profile_to_compare):
        return None
    tokens = set(profile_to_compare['freq'].keys())
    tokens.update(unknown_profile['freq'].keys())
    list_unknown_profile = []
    list_profile_to_compare = []
    for letter in tokens:
        list_profile_to_compare.append(profile_to_compare['freq'].get(letter, 0))
        list_unknown_profile.append(unknown_profile['freq'].get(letter, 0))
    return calculate_mse(list_profile_to_compare, list_unknown_profile)


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
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) or
            not isinstance(profile_2, dict)):
        return None
    mse_profile_1 = compare_profiles(unknown_profile, profile_1)
    mse_profile_2 = compare_profiles(unknown_profile, profile_2)
    if (isinstance(mse_profile_1, float)
            and isinstance(mse_profile_2, float)):
        if mse_profile_1 < mse_profile_2:
            return str(profile_1['name'])
        if mse_profile_2 < mse_profile_1:
            return str(profile_2['name'])
    return sorted([str(profile_1['name']), str(profile_2['name'])])[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, "r", encoding="utf-8") as json_file:
        language_profile = json.load(json_file)
    if not isinstance(language_profile, dict):
        return None
    return language_profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if (not isinstance(profile, dict) or 'name' not in profile
            or 'freq' not in profile or 'n_words' not in profile):
        return None
    new_freq = {}
    for key, value in profile['freq'].items():
        if key.lower() in new_freq:
            new_freq[key.lower()] += value / profile["n_words"][0]
        elif len(key) == 1:
            new_freq[key.lower()] = value / profile["n_words"][0]
    processed_profile = {'name': profile["name"], 'freq': new_freq}
    return processed_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    list_processed_profiles = []
    for paths in paths_to_profiles:
        language_profile = load_profile(paths)
        if isinstance(language_profile, dict):
            processed_profile = preprocess_profile(language_profile)
            if isinstance(processed_profile, dict):
                list_processed_profiles.append(processed_profile)
    return list_processed_profiles


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None
    list_mse = []
    for profile in known_profiles:
        if isinstance(profile, dict):
            list_mse.append((profile['name'], compare_profiles(unknown_profile, profile)))
    list_mse.sort(key=lambda a: (a[1], a[0]))
    return list_mse


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if isinstance(detections, list):
        for profile in detections:
            print(f'{profile[0]}: MSE {profile[1]:.5f}')
