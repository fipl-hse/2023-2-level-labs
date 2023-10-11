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
    new_text = ''
    for i in text:
        if i.isalpha():
            new_text += i
    tokens = list(new_text.lower())
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    dictionary = {}
    for i in tokens:
        if i in dictionary:
            dictionary[i] += 1 / len(tokens)
        else:
            dictionary[i] = 1 / len(tokens)
    return dictionary


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    lang_prof = {}
    lang_prof['name'] = language
    lang_prof['freq'] = calculate_frequencies(tokenize(text))
    return lang_prof


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list) or len(actual) != len(predicted):
        return None
    sum = 0
    for i in range(len(actual)):
        sum += (actual[i] - predicted[i]) ** 2
    mse = sum / len(actual)
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
    if (
        not isinstance(unknown_profile, dict) or
        not isinstance(profile_to_compare, dict) or
        ('name' or 'freq') not in unknown_profile or
        ('name' or 'freq') not in profile_to_compare
    ):
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
