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
    punctuation = '''!()-[]{};:'",<>./?@#$%^&*_~0123456789\\ '''
    return [el for el in text.lower() if el not in punctuation]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    frequency = {}
    for token in tokens:
        if not isinstance(token, str):
            return None
        frequency[token] = tokens.count(token) / len(tokens)
    return frequency


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
    if not isinstance(predicted, list) or not isinstance(actual, list) \
            or len(predicted) != len(actual):
        return None
    zipped_values = zip(predicted, actual)
    summation = 0
    for i in zipped_values:
        summation += (i[1] - i[0]) ** 2
    return summation / len(predicted)


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
    if not isinstance(unknown_profile, dict) or not isinstance(
            profile_to_compare, dict) or not ('name' or 'freq') in unknown_profile:
        return None
    tokens = set(profile_to_compare['freq'].keys())
    tokens.update(unknown_profile['freq'].keys())
    lang_to_compare = []
    unknown_lang = []
    for i in tokens:
        lang_to_compare.append(profile_to_compare['freq'].get(i, 0))
        unknown_lang.append(unknown_profile['freq'].get(i, 0))
    return calculate_mse(lang_to_compare, unknown_lang)


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

    if isinstance(mse_1, float) and isinstance(mse_2, float):
        if mse_1 < mse_2:
            return str(profile_1['name'])
        if mse_1 > mse_2:
            return str(profile_2['name'])
    return (sorted([str(profile_1['name']), str(profile_2['name'])]))[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return dict(data)


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not isinstance(profile, dict) or not ('freq' or 'name' or 'n_words') in profile:
        return None
    edited_profile = {'name': profile['name'], 'freq': {}}
    for token in profile['freq']:
        lowered_token = token.lower()
        frequency = profile['freq'][token]
        number_of_unigrams = profile['n_words'][0]
        if lowered_token in edited_profile['freq']:
            edited_profile['freq'][lowered_token] += frequency/number_of_unigrams
        elif len(token) == 1:
            edited_profile['freq'][lowered_token] = frequency/number_of_unigrams
    return edited_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    collected_profiles = []
    for path in paths_to_profiles:
        profile = load_profile(path)
        if isinstance(profile, dict) and isinstance(path, str):
            data = preprocess_profile(profile)
            if isinstance(data, dict):
                collected_profiles.append(data)
    if isinstance(collected_profiles, list):
        return collected_profiles
    return None


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
    detection = []
    for profile in known_profiles:
        tokens = set(unknown_profile['freq'].keys()) | set(profile['freq'].keys())
        unknown_lang = []
        known_lang = []
        for token in tokens:
            unknown_lang.append(unknown_profile['freq'].get(token, 0))
            known_lang.append(profile['freq'].get(token, 0))
        mse = calculate_mse(known_lang, unknown_lang)
        lang_mse = (profile['name'], mse)
        detection.append(lang_mse)
    return sorted(detection, key=lambda i: (i[1], i[0]))


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    for language in detections:
        print(f'{language[0]}: MSE {language[1]:.5f}')
