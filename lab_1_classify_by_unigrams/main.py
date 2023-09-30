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
    if not isinstance(text, str):
        return None
    text = text.lower()
    tokens = []
    for token in text:
        if token.isalpha():
            tokens.append(token)
    return tokens





def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    frequency_dict = {}
    tokens_number = len(tokens)
    for token in tokens:
        if type(token) != str:
            return None
        token_frequency = tokens.count(token)
        frequency_dict[token] = token_frequency / tokens_number
    return frequency_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    text = calculate_frequencies(tokenize(text))
    language_profile = {'name': language, 'freq': text}
    return language_profile



def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if len(predicted) != len(actual):
        return None
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    mse_sum = 0
    for index, element in enumerate(predicted):
        mse_sum += (element - actual[index]) ** 2
    mse = mse_sum / len(predicted)
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
    if 'name' and 'freq' not in unknown_profile and profile_to_compare:
        return None
    all_tokens = []
    unk_prof_freq = []
    prof_comp_freq = []
    for letter in unknown_profile['freq']:
        all_tokens.append(letter)
    for letter in profile_to_compare['freq']:
        if letter not in all_tokens:
            all_tokens.append(letter)
    for token in all_tokens:
        if token in unknown_profile['freq']:
            unk_prof_freq.append(unknown_profile['freq'][token])
        else:
            unk_prof_freq.append(0)
        if token in profile_to_compare['freq']:
            prof_comp_freq.append(profile_to_compare['freq'][token])
        else:
            prof_comp_freq.append(0)
    return calculate_mse(unk_prof_freq, prof_comp_freq)


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
    if not (isinstance(unknown_profile, dict)
            or isinstance(profile_1, dict)
            or isinstance(profile_2, dict)):
        return None
    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)
    if mse1 > mse2:
        return profile_2['name']
    elif mse1 < mse2:
        return profile_1['name']
    else:
        both_keys = [profile_1['name'], profile_2['name']]
        both_keys.sort()
        return both_keys[0]


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