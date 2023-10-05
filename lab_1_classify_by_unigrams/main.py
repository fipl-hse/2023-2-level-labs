"""
Lab 1
Language detection
"""
import json
import re


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if not isinstance(text, str):
        return None
    return [symbol for symbol in text.lower() if symbol.isalpha()]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list) or not all(isinstance(el, str) for el in tokens):
        return None
    freqs = {}
    for token in tokens:
        if token in freqs:
            freqs[token] += 1
        else:
            freqs[token] = 1
    for key, value in freqs.items():
        freqs[key] = value / len(tokens)
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
    freqs = calculate_frequencies(tokenize(text))
    if not freqs:
        return None
    return {"name": language, "freq": freqs}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list) \
            or (len(predicted) != len(actual)):
        return None
    dif = []
    for i, value in enumerate(actual):
        dif.append((value - predicted[i]) ** 2)
    return float(sum(dif) / len(dif))


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
            or (("name" or "freq") not in (unknown_profile.keys() or profile_to_compare.keys())):
        return None
    all_symbols = set(unknown_profile.get("freq").keys())\
        .union(set(profile_to_compare.get("freq").keys()))
    unknown_freq = []
    compare_freq = []
    for symbol in all_symbols:
        if symbol in unknown_profile.get("freq").keys():
            unknown_freq.append(unknown_profile.get("freq").get(symbol))
        else:
            unknown_freq.append(0)
        if symbol in profile_to_compare.get("freq").keys():
            compare_freq.append(profile_to_compare.get("freq").get(symbol))
        else:
            compare_freq.append(0)
    return calculate_mse(unknown_freq, compare_freq)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) \
            or not isinstance(profile_2, dict):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if not mse_1 or not mse_2:
        return None
    if mse_1 == mse_2:
        profs_to_sort = [profile_1.get("name"), profile_2.get("name")]
        profs_to_sort.sort()
        return str(profs_to_sort[0])
    if mse_1 < mse_2:
        return str(profile_1.get("name"))
    return str(profile_2.get("name"))


def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, "r", encoding="utf-8") as file_to_read:
        profile = json.load(file_to_read)
    if not isinstance(profile, dict):
        return None
    return profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not isinstance(profile, dict) or "freq" not in profile.keys() \
            or "name" not in profile.keys() or "n_words" not in profile.keys():
        return None
    profile_new = {"name": profile["name"], "freq": {}}
    for token, freq in profile.get("freq").items():
        if len(token) == 1 and re.search(r"\w", token):
            if token.lower() in profile_new["freq"]:
                profile_new["freq"][token.lower()] += freq / profile["n_words"][0]
            else:
                profile_new["freq"].update({token.lower(): freq / profile["n_words"][0]})
    return profile_new


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    profiles = []
    for path in paths_to_profiles:
        loaded_profile = load_profile(path)
        if not loaded_profile:
            return None
        preprocessed_profile = preprocess_profile(loaded_profile)
        if not preprocessed_profile:
            return None
        profiles.append(preprocessed_profile)
    return profiles


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
    mse_list = []
    for i, profile in enumerate(known_profiles):
        compare_prof_result = compare_profiles(unknown_profile, profile)
        if not compare_prof_result:
            return None
        mse_list.append((f"{profile.get('name')}", compare_prof_result))
    return sorted(mse_list, key=lambda x: (x[1], x[0]))


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    for detection in detections:
        print(f"{detection[0]}: MSE {detection[1]:.5f}")
