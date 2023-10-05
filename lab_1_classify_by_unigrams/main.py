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
    text = "".join(c for c in text if c.isalpha()).lower()
    return list(text)


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    res = all(isinstance(i, str) for i in tokens)
    if not res:
        return None
    dictionary = {}
    length = len(tokens)
    for i in tokens:
        if i not in dictionary:
            dictionary[i] = 0
        dictionary[i] += 1
    for i in dictionary:
        dictionary[i] = dictionary[i] / length
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
    dictionary = {
        "name": language,
        "freq": calculate_frequencies(tokenize(text))
    }
    return dictionary


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(actual) != len(predicted):
        return None
    if actual == predicted:
        return 0.0
    numerator = 0
    for i in range(len(actual)):
        numerator += (actual[i] - predicted[i])**2
    if len(predicted) == 0:
        return 1.0
    return numerator/len(predicted)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if 'name' not in unknown_profile or 'freq' not in unknown_profile:
        return None
    if 'name' not in profile_to_compare or 'freq' not in profile_to_compare:
        return None
    if (not isinstance(unknown_profile['freq'], dict) or
            not isinstance(profile_to_compare['freq'], dict)):
        return None
    for i in unknown_profile['freq']:
        if i not in profile_to_compare['freq']:
            profile_to_compare['freq'][i] = 0
    for j in profile_to_compare['freq']:
        if j not in unknown_profile['freq']:
            unknown_profile['freq'][j] = 0
    unknown_profile['freq'] = dict(sorted(unknown_profile['freq'].items()))
    profile_to_compare['freq'] = dict(sorted(profile_to_compare['freq'].items()))

    return calculate_mse(list(unknown_profile['freq'].values()), list(profile_to_compare['freq'].values()))


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
    if (not isinstance(unknown_profile, dict) or
        not isinstance(profile_1, dict) or
            not isinstance(profile_2, dict)):
        return None
    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)
    language1 = profile_1['name']
    language2 = profile_2['name']
    if mse1 > mse2:
        return language2
    if mse1 == mse2:
        return sorted([language1, language2])[0]
    return language1


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, 'r', encoding="utf-8") as f:
        profile = json.load(f)
    return dict(profile)


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if (not isinstance(profile, dict) or 'name' not in profile or
            'freq' not in profile or 'n_words'not in profile):
        return None
    dictionary = {'name': profile['name'], 'freq': {}}
    count_unigrams = profile['n_words'][0]
    for i in profile['freq']:
        if len(i) == 1:
            if i.lower() not in dictionary['freq']:
                dictionary['freq'][i.lower()] = profile['freq'][i]/count_unigrams

            else:
                dictionary['freq'][i.lower()] = dictionary['freq'][i.lower()] + profile['freq'][i] / count_unigrams

    return dictionary


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    collection = []
    for i in paths_to_profiles:
        collection += [preprocess_profile(load_profile(i))]
    return collection


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
    list_scores = []
    for profile in known_profiles:
        score = compare_profiles(unknown_profile, profile)
        t = tuple([profile['name'], score, 4])
        list_scores += [t]

    return sorted(list_scores, key=lambda a: a[1])


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if not isinstance(detections, list):
        return None
    for i in detections:
        print(f'{i[0]} MSE {i[1]:.5f}')
