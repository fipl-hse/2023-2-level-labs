"""
Lab 1
Language detection
"""
from typing import Any, Optional, Union


def tokenize(text: str) -> Optional[list[str]]:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """


def calculate_frequencies(tokens: Optional[list[str]]) -> Optional[dict[str, float]]:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """


def create_language_profile(language: str, text: str) -> Optional[dict[str, Union[str, dict[str, float]]]]:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """


def calculate_mse(predicted: list, actual: list) -> Optional[float]:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """


def compare_profiles(
    unknown_profile: dict[str, Union[str, dict[str, float]]],
    profile_to_compare: dict[str, Union[str, dict[str, float]]],
) -> Optional[float]:
    """
    Compares profiles and calculates the distance using symbols
    :param unknown_profile: a dictionary of an unknown profile
    :param profile_to_compare: a dictionary of a profile to compare the unknown profile to
    :return: the distance between the profiles
    """


def detect_language(
    unknown_profile: dict[str, Union[str, dict[str, float]]],
    profile_1: dict[str, Union[str, dict[str, float]]],
    profile_2: dict[str, Union[str, dict[str, float]]],
) -> Optional[Any]:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """


def load_profile(path_to_file: str) -> Any:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """


def preprocess_profile(profile: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """


def collect_profiles(paths_to_profiles: list) -> Optional[list[Optional[dict[str, Union[str, dict[str, float]]]]]]:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """


def detect_language_advanced(
    unknown_profile: dict[str, Union[str, dict[str, float]]], known_profiles: list
) -> Optional[list]:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """


def print_report(detections: list[list[Union[str, float]]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
