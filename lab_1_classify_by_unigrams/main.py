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

    tokens = [character for character in text.lower() if character.isalpha()]

    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list) or not all(isinstance(char, str) for char in tokens):
        return None

    text_freq = {}
    for character in tokens:
        if character not in text_freq:
            text_freq[character] = 0
        text_freq[character] += 1/len(tokens)

    return text_freq


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
    text_freq = calculate_frequencies(tokens)

    return {'name': language, 'freq': text_freq}


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

    mse = 0
    for index in range(len(predicted)):
        mse += (predicted[index] - actual[index])**2

    mse /= len(predicted)
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
            or ('name' or 'freq') not in unknown_profile \
            or ('name' or 'freq') not in profile_to_compare:
        return None

    mutual_profile = {}
    mutual_profile.update(profile_to_compare['freq'])
    mutual_profile.update(unknown_profile['freq'])
    mutual_characters = [[], []]
    for character in mutual_profile:
        mutual_characters[0].append(0)
        mutual_characters[1].append(0)

        if character in unknown_profile['freq']:
            mutual_characters[0][-1] = unknown_profile['freq'][character]
        if character in profile_to_compare['freq']:
            mutual_characters[1][-1] = profile_to_compare['freq'][character]

    if len(mutual_characters[0]) == len(mutual_characters[1]) == 0:
        return 1.
    return calculate_mse(mutual_characters[0], mutual_characters[1])


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
    if not all(isinstance(one_profile, dict) for one_profile in [unknown_profile, profile_1, profile_2]):
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)

    if mse_1 > mse_2:
        return profile_2['name']
    elif mse_1 < mse_2:
        return profile_1['name']
    return sorted([profile_1['name'], profile_2['name']])[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None

    with open(path_to_file, 'r', encoding='utf-8') as file:
        return json.load(file)


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not isinstance(profile, dict) or 'name' not in profile \
            or 'n_words' not in profile or 'freq' not in profile:
        return None

    perfect_profile = {'name': profile['name'],
                       'freq': {}}

    for sequence in profile['freq']:
        if sequence.lower() in perfect_profile['freq']:
            perfect_profile['freq'][sequence.lower()] += \
                profile['freq'][sequence] / profile['n_words'][0]
        elif len(sequence) == 1:
            perfect_profile['freq'][sequence.lower()] = \
                profile['freq'][sequence] / profile['n_words'][0]

    return perfect_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None

    loaded_profiles = []
    for path in paths_to_profiles:
        loaded_profiles.append(load_profile(path))

    preprocessed_profiles = []
    for profile in loaded_profiles:
        preprocessed_profiles.append(preprocess_profile(profile))

    return preprocessed_profiles


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if not isinstance(unknown_profile, dict) or \
            not isinstance(known_profiles, list):
        return None

    distances = []
    for profile in known_profiles:
        distances.append((profile['name'], compare_profiles(unknown_profile, profile)))

    distances.sort(key=lambda x: x[1])
    return distances


def print_report(detections: list[list[str | float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    for language in detections:
        print(f'{language[0]}: MSE {language[1]:.5f}')
