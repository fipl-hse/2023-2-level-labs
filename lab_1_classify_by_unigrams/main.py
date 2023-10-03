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

    if not all(isinstance(token, str) for token in tokens):
        return None

    return {token: tokens.count(token) / len(tokens) for token in tokens}


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None

    frequencies = calculate_frequencies(tokenize(text))
    if not isinstance(frequencies, dict):
        return None

    return {'name': language, 'freq': frequencies}


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

    sum_difference = 0
    for freq_value in zip(predicted, actual):
        sum_difference += (freq_value[0] - freq_value[1]) ** 2

    return sum_difference / len(predicted)


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
        isinstance(unknown_profile, dict) and 'name' in unknown_profile
        and 'freq' in unknown_profile and isinstance(profile_to_compare, dict)
        and 'name' in profile_to_compare and 'freq' in profile_to_compare
    ):
        return None
    all_tokens = set(unknown_profile['freq'].keys()) | set(profile_to_compare['freq'].keys())
    lang1_freq = []
    lang2_freq = []
    for token in all_tokens:
        lang1_freq.append(unknown_profile['freq'].get(token, 0))
        lang2_freq.append(profile_to_compare['freq'].get(token, 0))

    return calculate_mse(lang1_freq, lang2_freq)


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]]
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
            and isinstance(profile_1, dict) and isinstance(profile_2, dict)
    ):
        return None

    metrics = {
        profile_1['name']: compare_profiles(unknown_profile, profile_1),
        profile_2['name']: compare_profiles(unknown_profile, profile_2),
    }

    if not all(isinstance(metric, float) for metric in metrics.values()):
        return None

    if isinstance(metrics, dict):
        if metrics[profile_1['name']] == metrics[profile_2['name']]:
            return sorted(metrics.keys())[0]
        return min(metrics, key=metrics.get)

    return None


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None

    with open(path_to_file, "r", encoding="utf-8") as file:
        profile = json.load(file)

    return dict(profile)


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not (
            isinstance(profile, dict) and 'name' in profile
            and 'freq' in profile and 'n_words' in profile
    ):
        return None

    preprocessed_profile = {'name': profile['name'], 'freq': {}}
    unigrams = profile['n_words'][0]
    for token in profile['freq']:
        if token.lower() in preprocessed_profile['freq']:
            preprocessed_profile['freq'][token.lower()] += \
                profile['freq'][token] / profile['n_words'][0]
        elif len(token) == 1:
            token_lower = (token.lower())
            preprocessed_profile['freq'][token_lower] = profile['freq'][token] / unigrams

    return preprocessed_profile


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

    preprocess_profiles_list = []
    for path in paths_to_profiles:
        profile = load_profile(path)
        if isinstance(profile, dict):
            preprocessed_profile = preprocess_profile(profile)
            preprocess_profiles_list.append(preprocessed_profile)

    if (
            isinstance(preprocess_profiles_list, list)
            and all(isinstance(profile, dict) for profile in preprocess_profiles_list)
    ):
        return preprocess_profiles_list

    return None


def detect_language_advanced(
        unknown_profile: dict[str, str | dict[str, float]],
        known_profiles: list
) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if not (isinstance(unknown_profile, dict) and isinstance(known_profiles, list)):
        return None

    score_list = []
    for profile in known_profiles:
        all_tokens = set(profile["freq"].keys()) | set(unknown_profile["freq"].keys())
        unknown_freq = []
        known_freq = []
        for token in all_tokens:
            unknown_freq.append(unknown_profile["freq"].get(token, 0))
            known_freq.append(profile["freq"].get(token, 0))
        score_list.append((profile["name"], calculate_mse(unknown_freq, known_freq)))
    score_list = sorted(score_list, key=lambda score: (score[1], score[0]))

    if isinstance(score_list, list):
        return score_list

    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    for score in detections:
        score_val = f"{score[1]:.5f}"
        print(f"{score[0]}: MSE {score_val}")
