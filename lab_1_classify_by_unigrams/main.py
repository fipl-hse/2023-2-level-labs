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
    else:
        text_lower = text.lower()
    
    for symbol in text_lower:
        if not symbol.isalpha():
            text_lower = text_lower.replace(symbol, '')
    
    tokens = list(text_lower)

    return tokens


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
        
    tokens_frequency_dict = {}
    total_tokens = len(tokens)

    for token in tokens:
        if token in tokens_frequency_dict:
            tokens_frequency_dict[token] += 1
        else:
            tokens_frequency_dict[token] = 1

    for token in tokens_frequency_dict:
        tokens_frequency_dict[token] /= total_tokens
    
    return tokens_frequency_dict


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
    tokens_frequency_dict = calculate_frequencies(tokens)
    
    profile = {
        'name': language,
        'freq': tokens_frequency_dict
    }

    return profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """

    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    
    squared_differences = [(true - pred) ** 2 for true, pred in zip(actual, predicted)]
    mse = sum(squared_differences) / len(predicted)
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

    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if 'name' not in unknown_profile or 'name' not in profile_to_compare or \
       'freq' not in unknown_profile or 'freq' not in profile_to_compare:
        return None
    
    tokens1 = set(unknown_profile['freq'].keys())
    tokens2 = set(profile_to_compare['freq'].keys())

    all_tokens = tokens1.union(tokens2)

    actual = []
    predicted = []
    for token in all_tokens:
        actual.append(unknown_profile['freq'].get(token, 0))
        predicted.append(profile_to_compare['freq'].get(token, 0))

    mse = calculate_mse(predicted, actual)

    return round(mse, 3)


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

    if not isinstance(unknown_profile, dict) or \
       not isinstance(profile_1, dict) or \
       not isinstance(profile_2, dict):
        return None
    
    languages = list(profile_1.keys())

    if set(languages) != set(profile_2.keys()) or \
       set(languages) != set(unknown_profile.keys()):
        return None
    
    mse_lang_1 = compare_profiles(unknown_profile, profile_1)
    mse_lang_2 = compare_profiles(unknown_profile, profile_2)

    if mse_lang_1 is None or mse_lang_2 is None:
        return None

    if mse_lang_1 < mse_lang_2:
        return profile_1['name']
    elif mse_lang_1 > mse_lang_2:
        return profile_2['name']
    else:
        return sorted([profile_1['name'], profile_2['name']])[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """

    if not isinstance(path_to_file, str):
        return None
    
    try:
        with open(path_to_file, 'r') as file:
            language_profile = json.load(file)
            return language_profile
    except FileNotFoundError:
        return None


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """

    if not isinstance(profile, dict) or \
       not isinstance(profile['n_words'], list) or \
       profile.get('freq') is None or \
       profile.get('name') is None or \
       profile.get('n_words') is None:
        return None
    
    freq = profile['freq']
    unigrams = {}
    total_count = profile['n_words'][0]
    
    for token, count in freq.items():
        if len(token) == 1 and token.isalpha():
            unigram = token.lower()
            if unigram in unigrams:
                unigrams[unigram] += count
            else:
                unigrams[unigram] = count
    
    processed_profile = {
        'name': profile['name'],
        'freq': {unigram: count/total_count for unigram, count in unigrams.items() if count > 0}
    }

    return processed_profile


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
