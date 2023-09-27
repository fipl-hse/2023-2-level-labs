"""
Lab 1
Language detection
"""
def tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None
    else:
        text = text.lower()
        cleaned_text = ""
        for symbol in text:
            if symbol.isalpha() and symbol != " ":
                cleaned_text += symbol
        tokens = list(cleaned_text)
        return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
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
    dict_language_profile = {"name": language, "freq": calculate_frequencies(tokenize(text))}
    return dict_language_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    count_actual = len(actual)
    count_predicted = len(predicted)
    summ_values = 0
    if isinstance(actual, list) and isinstance(predicted, list) and count_actual == count_predicted:
        squared_difference = [(actual_value - predicted_value)**2 for actual_value, predicted_value in zip(actual,predicted)]
        for value in squared_difference:
            summ_values += value
        mse = round(summ_values / count_actual, 3)
        return mse
    else:
        return None


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
    if isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict):
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
    else:
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
    if isinstance(unknown_profile, dict) and isinstance(profile_1, dict) and isinstance(profile_2, dict):
        mse_profile_1 = compare_profiles(unknown_profile, profile_1)
        mse_profile_2 = compare_profiles(unknown_profile, profile_2)
        if mse_profile_1 < mse_profile_2:
            return profile_1['name']
        elif mse_profile_2 < mse_profile_1:
            return profile_2['name']
        else:
            str_name_language = sorted(profile_1['name'] + profile_2['name'])
            first_name = str_name_language[0]
            return first_name
    else:
        return None


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
