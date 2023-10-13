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

    return [token.lower() for token in text if token.isalpha()]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
<<<<<<< HEAD

    if not isinstance(tokens, list):
        return None

    length = len(tokens)
    freq_dict = {}
    for token in set(tokens):
        if not isinstance(token, str):
            return None

        freq_dict[token] = tokens.count(token) / length

    return freq_dict

=======
    if not (
            isinstance(tokens, list) and all(isinstance(el, str) for el in tokens)
    ):
        return None

    frequency_dict = {el: (tokens.count(el) / len(tokens)) for el in tokens}
    return frequency_dict
>>>>>>> 65edb31c98e5022471da1958587681a895d1b48e


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not (
            isinstance(language, str) and isinstance(text, str)
    ):
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
            isinstance(predicted, list)
            and isinstance(actual, list)
            and len(predicted) == len(actual)
    ):
        return None

    sum_diff = 0
    for freq_value in zip(predicted, actual):
        sum_diff += (freq_value[0] - freq_value[1]) ** 2
    mse = sum_diff / len(predicted)
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
    if not (
            isinstance(unknown_profile, dict)
            and isinstance(profile_to_compare, dict)
            and 'name' in unknown_profile
            and 'freq' in unknown_profile
            and 'name' in profile_to_compare
            and 'freq' in profile_to_compare
    ):
        return None

    unknown_tokens = set(unknown_profile.get('freq').keys())
    compare_tokens = set(profile_to_compare.get('freq').keys())
    all_tokens = unknown_tokens | compare_tokens
    unknown_freq = []
    compare_freq = []
    for token in all_tokens:
        unknown_freq.append(unknown_profile['freq'].get(token, 0))
        compare_freq.append(profile_to_compare['freq'].get(token, 0))

    calculated_mse = calculate_mse(unknown_freq, compare_freq)
    return calculated_mse


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
    if not (
            isinstance(unknown_profile, dict)
            and isinstance(profile_1, dict)
            and isinstance(profile_2, dict)
    ):
        return None

    profile_1_metric = compare_profiles(unknown_profile, profile_1)
    profile_2_metric = compare_profiles(unknown_profile, profile_2)

    if not (isinstance(profile_1_metric, float)
            and isinstance(profile_2_metric, float)
    ):
        return None

    if profile_1_metric > profile_2_metric:
        return str(profile_2['name'])
    if profile_1_metric < profile_2_metric:
        return str(profile_1['name'])

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
    if not (isinstance(profile, dict)
        and 'name' in profile
        and 'freq' in profile
        and 'n_words' in profile
    ):
        return None

    unigram_profile = {'name': profile['name'], 'freq': {}}
    for token in profile['freq']:
        if token.lower() in unigram_profile['freq']:
            unigram_profile['freq'][token.lower()] += profile['freq'][token] / profile['n_words'][0]
        elif len(token) == 1 and (token.isalpha() or token == '²'):
            unigram_profile['freq'][token.lower()] = profile['freq'][token] / profile['n_words'][0]

    return unigram_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None

    if not all(isinstance(path, str) for path in paths_to_profiles):
        return None

    profiles = []
    for path in paths_to_profiles:
        loaded_profile = load_profile(path)
        if not loaded_profile:
            return None

        preprocessed_profile = preprocess_profile(loaded_profile)
        if not preprocessed_profile:
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
    if not (isinstance(unknown_profile, dict)
            and isinstance(known_profiles, list)
    ):
        return None

    detected_language = [(profile['name'], compare_profiles(profile, unknown_profile))
                         for profile in known_profiles]
    detected_language = sorted(detected_language, key=lambda x: (x[1], x[0]))

    if not isinstance(detected_language, list):
        return None

    return detected_language


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if isinstance(detections, list):
        for detection in detections:
            print(f'{detection[0]}: MSE {detection[1]:.5f}')
