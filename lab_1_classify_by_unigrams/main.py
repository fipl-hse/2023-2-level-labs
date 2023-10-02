"""
Lab 1
Language detection turi ip ip
"""


def tokenize(text: str) -> list[str] | None:
    if type(text) is str:
        tokenized = []
        for token in text:
            if token.isalnum() and not token.isnumeric():
                tokenized.append(token.lower())
        return tokenized
    else:
        return None

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if type(tokens) is list:
        dict_instances = {}
        dict_relative_frequencies = {}
        total_number = 0
        for token in tokens:
            if token in dict_instances:
                dict_instances[token] += 1
                total_number += 1
            else:
                dict_instances[token] = 1
                total_number += 1
        for key, value in dict_instances.items():
            dict_relative_frequencies[key] = value / total_number
        return dict_relative_frequencies
    else:
        return None

def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if type(language) is str and type(text) is str:
        language_profile = {}
        language_profile["name"] = language
        language_profile["freq"] = calculate_frequencies(tokenize(text))
        return language_profile
    else:
        return None

def calculate_mse(predicted: list, actual: list) -> float | None:
    if type(predicted) is list and type(actual) is list:
        list_sum = []
        list_squared = []
        n = 0
        summa = 0
        for i in predicted:
            list_sum.append(i - actual[n])
            n += 1
        for i in list_sum:
            list_squared.append(i ** 2)
        for i in list_squared:
            summa += i
        return summa / n
    else:
        return None

def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    if type(unknown_profile) is dict and type(profile_to_compare) is dict and "name" in unknown_profile and "name" in profile_to_compare and "freq" in unknown_profile and "freq" in profile_to_compare:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        unknown_profile_list = []
        profile_to_compare_list = []
        for i in alphabet:
            if i in unknown_profile["freq"] and i in profile_to_compare["freq"]:
                unknown_profile_list.append(unknown_profile["freq"][i])
                profile_to_compare_list.append(profile_to_compare["freq"][i])
        return calculate_mse(unknown_profile_list, profile_to_compare_list)
    else:
        return None

def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    if type(unknown_profile) is dict and type(profile_1) is dict and type(profile_2) is dict:
        first_pair = compare_profiles(
            unknown_profile,
            profile_1
        )
        second_pair = compare_profiles(
            unknown_profile,
            profile_2
        )
        if first_pair < second_pair:
            return profile_1["name"]
        if second_pair < first_pair:
            return profile_2["name"]
        if first_pair == second_pair:
            if profile_1["name"] > profile_2["name"]:
                return profile_2["name"]
            if profile_1["name"] < profile_2["name"]:
                return profile_1["name"]
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
