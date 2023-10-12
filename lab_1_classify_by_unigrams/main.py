"""
Lab 1
Language detection
"""

def tokenize(text: str) -> list[str] | None:
    punc = '''0123456789!()-[]{};:'"\\,<>./?@#$%^&*_~ '''
    if type(text) == str:
        text1 = text.lower()
        for symbol in text1:
            if symbol in punc:  # проверка, не является ли символ пробелом или знаокм препинания
                text1 = text1.replace(symbol, '')  # удаление пробелов и знаков препинания
        tokens = [symbol for symbol in text1]
        return tokens
    else:
        return None

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if type (tokens) == list:
        dict = {}
        for token in tokens:
            if type(token) == str:
                dict[token] = tokens.count(token)/len(tokens)
            else:
                return None
        return dict
    else:
        return None

def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if type(language) == str and type(text) == str:
        freq_dict = calculate_frequencies(tokenize(text))
        dict = {
            "name": language,
            "freq": freq_dict
        }
        return dict
    else:
        return None

#create_language_profile(en, en_text)

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
