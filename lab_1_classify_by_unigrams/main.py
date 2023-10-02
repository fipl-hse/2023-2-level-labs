"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if isinstance(text, str) == False:
        return None
    tokens = [t for t in text.lower() if (t.isalpha() and t != 'º')]
    return tokens

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if isinstance(tokens, list) == False:
        return None
    for token in tokens:
        if isinstance(token, str) == False:
            return None
    dict_tokens = {}
    all_tokens = 0
    for token in tokens:
        all_tokens += 1
        num_token = tokens.count(token)
        dict_tokens[token] = num_token
    dict_freq = {}
    for key, value in dict_tokens.items():
        freq = value/all_tokens
        dict_freq[key] = freq
    return dict_freq


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if isinstance(language, str) == False or isinstance(text, str) == False:
        return None
    lang_profile = {}
    lang_profile['name'] = language
    lang_profile['freq'] = calculate_frequencies(tokenize(text))
    return lang_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if isinstance(predicted, list) == False or isinstance(actual, list) == False or len(predicted) != len(actual):
        return None
    sum_diff = 0
    for i in range(0, len(predicted)):
        sum_diff += (actual[i] - predicted[i]) ** 2
    mse = sum_diff / len (predicted)
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
    if isinstance(unknown_profile, dict) == False or isinstance(profile_to_compare, dict) == False:
        return None
    if 'name' not in unknown_profile or  'freq' not in unknown_profile:
        return None
    if 'name' not in profile_to_compare or 'freq' not in profile_to_compare:
        return None

    sorted_profile_to_compare = dict(sorted(profile_to_compare.items()))
    sorted_unknown_profile = dict(sorted(unknown_profile.items()))

    predicted_tokens = sorted_profile_to_compare.get('freq')
    actual_tokens = sorted_unknown_profile.get('freq')

    for key in predicted_tokens.keys():
        if key not in actual_tokens:
            actual_tokens[key] = 0

    for key in actual_tokens.keys():
        if key not in predicted_tokens:
            predicted_tokens[key] = 0

    predicted = []
    for value in predicted_tokens.values():
        predicted.append(value)

    actual = []
    for value in actual_tokens.values():
        actual.append(value)

    mse = calculate_mse(predicted, actual)
    mse = round(mse, 3)
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
    if isinstance(unknown_profile, dict) == False:
        return None
    if isinstance(profile_1, dict) == False or isinstance(profile_2, dict) == False:
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 == mse_2:
        names = []
        names.append(profile_1['name'])
        names.append(profile_2['name'])
        names.sort()
        language =  names[0]
    elif mse_1 < mse_2:
        language = profile_1['name']
    else:
        language = profile_2['name']
    return language


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """


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
