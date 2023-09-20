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
    if isinstance(text, str):
        tokens = []
        for symbol in text:
            if symbol.isalpha():
                tokens.append(symbol.lower())
        return tokens
    else:
        return None


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    check_relevance = False
    if isinstance(tokens, list):
        for element in tokens:
            if isinstance(element, str):
                check_relevance = True
    if check_relevance:
        frequency = {}
        for token in tokens:
            if token not in frequency:
                frequency[token] = tokens.count(token) / len(tokens)
        return frequency
    else:
        return None


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if isinstance(language, str) and isinstance(text, str):
        language_profile = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
        return language_profile
    else:
        return None


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if isinstance(predicted, list) and isinstance(actual, list) and len(predicted) == len(actual):
        sum_difference = 0
        for i in range(len(predicted)):
            sum_difference += (predicted[i] - actual[i]) ** 2
        mse = sum_difference / len(predicted)
        return mse
    else:
        return None


def validate_input(dict1):
    if isinstance(dict1, dict) and 'name' in dict1 and 'freq' in dict1:
        return True
    else:
        return False


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
    if validate_input(unknown_profile) and validate_input(profile_to_compare):
        all_tokens = set(unknown_profile['freq'].keys()) | set(profile_to_compare['freq'].keys())
        lang1_freq = []
        lang2_freq = []
        for token in all_tokens:
            lang1_freq.append(unknown_profile['freq'].get(token, 0))
            lang2_freq.append(profile_to_compare['freq'].get(token, 0))
        return calculate_mse(lang1_freq, lang2_freq)
    else:
        return None


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
    if isinstance(unknown_profile, dict) and isinstance(profile_1, dict) and isinstance(profile_2, dict):
        metrics = {
            profile_1['name']: compare_profiles(unknown_profile, profile_1),
            profile_2['name']: compare_profiles(unknown_profile, profile_2)
        }
        if metrics[profile_1['name']] == metrics[profile_2['name']]:
            return sorted(metrics)[0]
        else:
            return min(metrics, key=metrics.get)
    else:
        return None


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if isinstance(path_to_file, str):
        with open(path_to_file,  'r', encoding='utf-8') as file:
            profile = json.load(file)
        return profile
    else:
        return None


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if isinstance(profile, dict) and 'name' in profile and 'freq' in profile and 'n_words' in profile:
        preprocessed_profile = {'name': profile['name'], 'freq': {}}
        for token in profile['freq']:
            if len(token.strip()) == 1 and token.isalpha():
                preprocessed_profile['freq'][(token.strip()).lower()] = profile['freq'][token] / profile['n_words'][0]
        return preprocessed_profile
    else:
        return None


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if isinstance(paths_to_profiles, list):
        preprocess_profile_list = []
        for path in paths_to_profiles:
            preprocess_profile_list.append(preprocess_profile(load_profile(path)))
        return preprocess_profile_list
    else:
        return None


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if isinstance(unknown_profile, dict) and isinstance(known_profiles, list):
        score_list = []
        for profile in known_profiles:
            all_tokens = set(profile['freq'].keys()) | set(unknown_profile['freq'].keys())
            unknown_freq = []
            known_freq = []
            for token in all_tokens:
                unknown_freq.append(unknown_profile['freq'].get(token, 0))
                known_freq.append(profile['freq'].get(token, 0))
            score_list.append((profile['name'], calculate_mse(unknown_freq, known_freq)))
        score_list = sorted(score_list, key=lambda score: (score[1], score[0]))
        return score_list
    else:
        return None


def print_report(detections: list[list[str | float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    for score in detections:
        score_val = f'{score[1]:.5f}'
        print(f'{score[0]}: MSE {score_val}')
