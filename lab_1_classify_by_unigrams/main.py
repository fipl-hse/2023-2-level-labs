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

    tokens = []

    for symbol in text:
        if symbol.isalpha():
            tokens.append(symbol.lower())

    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None

    tokens_frequency_dict = {}
    total_tokens = len(tokens)

    for token in tokens:
        if not isinstance(token, str):
            return None
        tokens_frequency_dict[token] = tokens_frequency_dict.get(token, 0) + 1

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

    if not isinstance(tokens_frequency_dict, dict):
        return None

    return {'name': language, 'freq': tokens_frequency_dict}


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

    mse = 0
    squared_differences = [(true - pred) ** 2 for true, pred in zip(actual, predicted)]
    mse += sum(squared_differences) / len(predicted)

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
    if ('name' not in unknown_profile
        or 'name' not in profile_to_compare
        or 'freq' not in unknown_profile
        or 'freq' not in profile_to_compare):
        return None

    tokens_unknown = set(unknown_profile['freq'].keys())
    tokens_to_compare = set(profile_to_compare['freq'].keys())
    total_tokens = tokens_unknown.union(tokens_to_compare)
    actual = []
    predicted = []

    for token in total_tokens:
        actual.append(unknown_profile['freq'].get(token, 0))
        predicted.append(profile_to_compare['freq'].get(token, 0))

    mse = calculate_mse(predicted, actual)

    return mse


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
    if (not isinstance(unknown_profile, dict)
        or not isinstance(profile_1, dict)
        or not isinstance(profile_2, dict)):
        return None

    languages = list(profile_1.keys())

    if set(languages) != set(profile_2.keys()) or \
       set(languages) != set(unknown_profile.keys()):
        return None

    mse_lang_1 = compare_profiles(unknown_profile, profile_1)
    mse_lang_2 = compare_profiles(unknown_profile, profile_2)
    lang_1 = str(profile_1['name'])
    lang_2 = str(profile_2['name'])

    if mse_lang_1 < mse_lang_2:
        return lang_1
    if mse_lang_1 > mse_lang_2:
        return lang_2
    if mse_lang_1 == mse_lang_2:
        return sorted([lang_1, lang_2])[0]

    return None


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None

    try:
        with open(path_to_file, 'r', encoding="utf-8") as file:
            language_profile = json.load(file)
            if not isinstance(language_profile, dict):
                raise ValueError("Invalid language profile format. Expected a dictionary.")
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
    if (not isinstance(profile, dict)
        or not isinstance(profile['n_words'], list)
        or profile.get('freq') is None
        or profile.get('name') is None
        or profile.get('n_words') is None):
        return None

    processed_profile = {'name': profile['name'], 'freq': {}}
    freq_raw = profile['freq']
    freq_processed = processed_profile['freq']

    for token in freq_raw:
        if token.lower() in freq_processed:
            freq_processed[token.lower()] += freq_raw[token] / profile['n_words'][0]
        elif len(token) == 1:
            freq_processed[token.lower()] = freq_raw[token] / profile['n_words'][0]

    return processed_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None

    processed_profiles = []
    for path in paths_to_profiles:
        profile = load_profile(path)
        if profile is None:
            continue
        processed_profile = preprocess_profile(profile)
        if processed_profile is None:
            continue
        processed_profiles.append(processed_profile)

    return processed_profiles


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

    result = []
    for profile in known_profiles:
        result.append((profile['name'], compare_profiles(unknown_profile, profile)))

    result.sort(key=lambda score: (score[1], score[0]))

    return result


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if isinstance(detections, list):
        for lang, score in detections:
            print(f'{lang}: MSE {round(score, 5)}')
