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
    if isinstance(text, str):
        text = [i for i in text.lower() if i.isalpha()]
        return text
    else:
        return None

    tokenize(text)

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    tokens = ['v', 'o', 'n', 'f', 'r', 'i', 's', 'c', 'h', 'n', 'o', 't', 'e', 'd', 's', 'o', 'm', 'e', 't', 'h', 'i',
              'n', 'g', 'f', 'u', 'r', 't', 'h', 'e', 'r', 'w', 'h', 'e', 'n', 't', 'h', 'e', 's', 'c', 'o', 'u', 't',
              'b', 'e', 'e', 's', 'c', 'a', 'm', 'e', 'h', 'o', 'm', 'e', 't', 'o', 't', 'e', 'l', 'l', 't', 'h', 'e',
              'i', 'r', 's', 'i', 's', 't', 'e', 'r', 's', 'a', 'b', 'o', 'u', 't', 't', 'h', 'e', 'f', 'o', 'o', 'd',
              's', 'o', 'u', 'r', 'c', 'e', 's', 'o', 'm', 'e', 't', 'i', 'm', 'e', 's', 't', 'h', 'e', 'y', 'w', 'o',
              'u', 'l', 'd', 'd', 'a', 'n', 'c', 'e', 'o', 'u', 't', 's', 'i', 'd', 'e', 'o', 'n', 't', 'h', 'e', 'h',
              'o', 'r', 'i', 'z', 'o', 'n', 't', 'a', 'l', 'e', 'n', 't', 'r', 'a', 'n', 'c', 'e', 'p', 'l', 'a', 't',
              'f', 'o', 'r', 'm', 'o', 'f', 't', 'h', 'e', 'h', 'i', 'v', 'e', 'a', 'n', 'd', 's', 'o', 'm', 'e', 't',
              'i', 'm', 'e', 's', 'o', 'n', 't', 'h', 'e', 'v', 'e', 'r', 't', 'i', 'c', 'a', 'l', 'w', 'a', 'l', 'l',
              'i', 'n', 's', 'i', 'd', 'e', 'a', 'n', 'd', 'd', 'e', 'p', 'e', 'n', 'd', 'i', 'n', 'g', 'o', 'n', 'w',
              'h', 'e', 'r', 'e', 't', 'h', 'e', 'y', 'd', 'a', 'n', 'c', 'e', 'd', 't', 'h', 'e', 's', 't', 'r', 'a',
              'i', 'g', 'h', 't', 'p', 'o', 'r', 't', 'i', 'o', 'n', 'o', 'f', 't', 'h', 'e', 'w', 'a', 'g', 'g', 'l',
              'e', 'd', 'a', 'n', 'c', 'e', 'w', 'o', 'u', 'l', 'd', 'p', 'o', 'i', 'n', 't', 'i', 'n', 'd', 'i', 'f',
              'f', 'e', 'r', 'e', 'n', 't', 'd', 'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 's', 't', 'h', 'e', 'o', 'u',
              't', 's', 'i', 'd', 'e', 'd', 'a', 'n', 'c', 'e', 'w', 'a', 's', 'f', 'a', 'i', 'r', 'l', 'y', 'e', 'a',
              's', 'y', 't', 'o', 'd', 'e', 'c', 'o', 'd', 'e', 't', 'h', 'e', 's', 't', 'r', 'a', 'i', 'g', 'h', 't',
              'p', 'o', 'r', 't', 'i', 'o', 'n', 'o', 'f', 't', 'h', 'e', 'd', 'a', 'n', 'c', 'e', 'p', 'o', 'i', 'n',
              't', 'e', 'd', 'd', 'i', 'r', 'e', 'c', 't', 'l', 'y', 't', 'o', 't', 'h', 'e', 'f', 'o', 'o', 'd', 's',
              'o', 'u', 'r', 'c', 'e', 's', 'o', 't', 'h', 'e', 'b', 'e', 'e', 's', 'w', 'o', 'u', 'l', 'd', 'm', 'e',
              'r', 'e', 'l', 'y', 'h', 'a', 'v', 'e', 't', 'o', 'd', 'e', 'c', 'o', 'd', 'e', 't', 'h', 'e', 'd', 'i',
              's', 't', 'a', 'n', 'c', 'e', 'm', 'e', 's', 's', 'a', 'g', 'e', 'a', 'n', 'd', 'f', 'l', 'y', 'o', 'f',
              'f', 'i', 'n', 't', 'h', 'a', 't', 'd', 'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 't', 'o', 'f', 'i', 'n',
              'd', 't', 'h', 'e', 'i', 'r', 'f', 'o', 'o', 'd', 'b', 'u', 't', 'b', 'y', 's', 't', 'u', 'd', 'y', 'i',
              'n', 'g', 't', 'h', 'e', 'd', 'a', 'n', 'c', 'e', 'o', 'n', 't', 'h', 'e', 'i', 'n', 'n', 'e', 'r', 'w',
              'a', 'l', 'l', 'o', 'f', 't', 'h', 'e', 'h', 'i', 'v', 'e', 'v', 'o', 'n', 'f', 'r', 'i', 's', 'c', 'h',
              'd', 'i', 's', 'c', 'o', 'v', 'e', 'r', 'e', 'd', 'a', 'r', 'e', 'm', 'a', 'r', 'k', 'a', 'b', 'l', 'e',
              'm', 'e', 't', 'h', 'o', 'd', 'w', 'h', 'i', 'c', 'h', 't', 'h', 'e', 'd', 'a', 'n', 'c', 'e', 'r', 'u',
              's', 'e', 'd', 't', 'o', 't', 'e', 'l', 'l', 'h', 'e', 'r', 's', 'i', 's', 't', 'e', 'r', 's', 't', 'h',
              'e', 'd', 'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 'o', 'f', 't', 'h', 'e', 'f', 'o', 'o', 'd', 'i', 'n',
              'r', 'e', 'l', 'a', 't', 'i', 'o', 'n', 't', 'o', 't', 'h', 'e', 's', 'u', 'n', 'w', 'h', 'e', 'n', 'i',
              'n', 's', 'i', 'd', 'e', 't', 'h', 'e', 'h', 'i', 'v', 'e', 't', 'h', 'e', 'd', 'a', 'n', 'c', 'e', 'r',
              'c', 'a', 'n', 'n', 'o', 't', 'u', 's', 'e', 't', 'h', 'e', 's', 'u', 'n', 's', 'o', 's', 'h', 'e', 'u',
              's', 'e', 's', 'g', 'r', 'a', 'v', 'i', 't', 'y', 'i', 'n', 's', 't', 'e', 'a', 'd', 't', 'h', 'e', 'd',
              'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 'o', 'f', 't', 'h', 'e', 's', 'u', 'n', 'i', 's', 'r', 'e', 'p',
              'r', 'e', 's', 'e', 'n', 't', 'e', 'd', 'b', 'y', 't', 'h', 'e', 't', 'o', 'p', 'o', 'f', 't', 'h', 'e',
              'h', 'i', 'v', 'e', 'w', 'a', 'l', 'l', 'i', 'f', 's', 'h', 'e', 'r', 'u', 'n', 's', 's', 't', 'r', 'a',
              'i', 'g', 'h', 't', 'u', 'p', 't', 'h', 'i', 's', 'm', 'e', 'a', 'n', 's', 't', 'h', 'a', 't', 't', 'h',
              'e', 'f', 'e', 'e', 'd', 'i', 'n', 'g', 'p', 'l', 'a', 'c', 'e', 'i', 's', 'i', 'n', 't', 'h', 'e', 's',
              'a', 'm', 'e', 'd', 'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 'a', 's', 't', 'h', 'e', 's', 'u', 'n', 'h',
              'o', 'w', 'e', 'v', 'e', 'r', 'i', 'f', 'f', 'o', 'r', 'e', 'x', 'a', 'm', 'p', 'l', 'e', 't', 'h', 'e',
              'f', 'e', 'e', 'd', 'i', 'n', 'g', 'p', 'l', 'a', 'c', 'e', 'i', 's', 'º', 't', 'o', 't', 'h', 'e', 'l',
              'e', 'f', 't', 'o', 'f', 't', 'h', 'e', 's', 'u', 'n', 't', 'h', 'e', 'n', 't', 'h', 'e', 'd', 'a', 'n',
              'c', 'e', 'r', 'w', 'o', 'u', 'l', 'd', 'r', 'u', 'n', 'º', 't', 'o', 't', 'h', 'e', 'l', 'e', 'f', 't',
              'o', 'f', 't', 'h', 'e', 'v', 'e', 'r', 't', 'i', 'c', 'a', 'l', 'l', 'i', 'n', 'e', 't', 'h', 'i', 's',
              'w', 'a', 's', 't', 'o', 'b', 'e', 't', 'h', 'e', 'f', 'i', 'r', 's', 't', 'o', 'f', 'v', 'o', 'n', 'f',
              'r', 'i', 's', 'c', 'h', 's', 'r', 'e', 'm', 'a', 'r', 'k', 'a', 'b', 'l', 'e', 'd', 'i', 's', 'c', 'o',
              'v', 'e', 'r', 'i', 'e', 's', 's', 'o', 'o', 'n', 'h', 'e', 'w', 'o', 'u', 'l', 'd', 'a', 'l', 's', 'o',
              'd', 'i', 's', 'c', 'o', 'v', 'e', 'r', 'a', 'n', 'u', 'm', 'b', 'e', 'r', 'o', 'f', 'o', 't', 'h', 'e',
              'r', 'r', 'e', 'm', 'a', 'r', 'k', 'a', 'b', 'l', 'e', 'f', 'a', 'c', 't', 's', 'a', 'b', 'o', 'u', 't',
              'h', 'o', 'w', 'b', 'e', 'e', 's', 'c', 'o', 'm', 'm', 'u', 'n', 'i', 'c', 'a', 't', 'e', 'a', 'n', 'd',
              'i', 'n', 'd', 'o', 'i', 'n', 'g', 's', 'o', 'r', 'e', 'v', 'o', 'l', 'u', 't', 'i', 'o', 'n', 'i', 's',
              'e', 't', 'h', 'e', 's', 't', 'u', 'd', 'y', 'o', 'f', 'a', 'n', 'i', 'm', 'a', 'l', 'b', 'e', 'h', 'a',
              'v', 'i', 'o', 'u', 'r', 'g', 'e', 'n', 'e', 'r', 'a', 'l', 'l', 'y']

    if isinstance(tokens, list):

        tokens_count = len(tokens)

        list_tokens = {}

        for i in tokens:
            if i in list_tokens:
                list_tokens[i] += 1
            else:
                list_tokens[i] = 1

        '''for i in list_tokens():
            tokens_value = list_tokens.value/tokens_count'''

        freq = {letter: value / tokens_count for letter, value in list_tokens.items()}

        return freq
    else:
        return None
    calculate_frequencies(tokens)



def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """


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
