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
    cleaned_text = []
    for symbol in text:
        if symbol.isalpha():
            cleaned_text.append(symbol.lower())
    return cleaned_text


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
    freqs = {}
    for token in tokens:
        if token in freqs:
            freqs[token] += 1
        else:
            freqs[token] = 1
    for token, freq in freqs.items():
        freqs[token] = freq / len(tokens)
    return freqs


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
        Creates a language profile
        :param language: a language
        :param text: a text
        :return: a dictionary with two keys – name, freq
        """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    dict_language_profile = {"name": language, "freq": calculate_frequencies(tokenize(text))}
    return dict_language_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(actual, list) or not isinstance(predicted, list) or len(actual) != len(predicted):
        return None
    summ_values = 0
    for i, predicted_value in enumerate(predicted):
        summ_values += (actual[i] - predicted_value)**2
    mse = round(summ_values / len(actual), 4)
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
    if (not isinstance(unknown_profile, dict) or
            not isinstance(profile_to_compare, dict) or
            'name' not in unknown_profile or
            'name' not in profile_to_compare):
        return None
    values_unknown_profile = unknown_profile['freq']
    values_profile_to_compare = profile_to_compare['freq']
    for letter in values_unknown_profile:
        if letter not in values_profile_to_compare:
            values_profile_to_compare[letter] = 0
    for letter in values_profile_to_compare:
        if letter not in values_unknown_profile:
            values_unknown_profile[letter] = 0
    sorted_unknown_profile = dict(sorted(values_unknown_profile.items()))
    sorted_profile_to_compare = dict(sorted(values_profile_to_compare.items()))
    list_unknown_profile = list(sorted_unknown_profile.values())
    list_profile_to_compare = list(sorted_profile_to_compare.values())
    profile_difference = calculate_mse(list_unknown_profile, list_profile_to_compare)
    return profile_difference


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
    mse_profile_1 = compare_profiles(unknown_profile, profile_1)
    mse_profile_2 = compare_profiles(unknown_profile, profile_2)
    if mse_profile_1 < mse_profile_2:
        return profile_1['name']
    if mse_profile_2 < mse_profile_1:
        return profile_2['name']
    else:
        str_name_language = sorted(profile_1['name'] + profile_2['name'])
        first_name = str_name_language[0]
        return first_name


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
