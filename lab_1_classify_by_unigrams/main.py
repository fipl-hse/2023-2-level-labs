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
    new_text = ""
    for symbol in text:
        if symbol.isalpha():
            new_text += symbol
    tokens = list(new_text)
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
    all_tokens = len(tokens)
    calc = {}.fromkeys(tokens, 0)
    for token in tokens:
        calc[token] += 1
    for key in calc:
        calc[key] /= all_tokens
    return calc


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """

    if not isinstance(language, str) or not isinstance(text, str):
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    lang_prof = {"name": language, "freq": freq_dict}
    return lang_prof


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
    difference_square = []
    quantity_of_meanings = len(predicted)
    for i in range(quantity_of_meanings):
        difference_square.append((actual[i] - predicted[i]) ** 2)
    mse = sum(difference_square) / quantity_of_meanings
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
    if not ('name' and 'freq') in unknown_profile or not ('name' and 'freq') in profile_to_compare:
        return None
    unknown = set(unknown_profile['freq'])
    comparable = set(profile_to_compare['freq'])
    set_of_tokens = unknown.union(comparable)
    unknown_freq = []
    to_compare_freq = []
    for token in set_of_tokens:
        unknown_freq.append(unknown_profile['freq'].get(token, 0))
        to_compare_freq.append(profile_to_compare['freq'].get(token, 0))
    mse_to_compare = calculate_mse(unknown_freq, to_compare_freq)
    return mse_to_compare


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]]
) -> str | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """
    if isinstance(
            unknown_profile, dict) or isinstance(profile_1, dict
                                                 ) or isinstance(profile_2, dict
                                                                 ):
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(mse_1, float) or not isinstance(mse_2, float):
        return None
    if mse_1 < mse_2:
        return str(profile_1['name'])
    if mse_2 < mse_1:
        return str(profile_2['name'])
    return str([profile_1['name'], profile_2['name']].sort())
