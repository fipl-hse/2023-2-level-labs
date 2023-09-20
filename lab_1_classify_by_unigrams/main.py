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
    text = '''Von Frisch noted something further. When the scout bees came home to tell their
sisters about the food source, sometimes they would dance outside on the horizontal
entrance platform of the hive, and sometimes on the vertical wall inside. And,
depending on where they danced, the straight portion of the waggle dance would
point in different directions. The outside dance was fairly easy to decode: the straight
portion of the dance pointed directly to the food source, so the bees would merely
have to decode the distance message and fly off in that direction to find their food.
But by studying the dance on the inner wall of the hive, von Frisch discovered a
remarkable method which the dancer used to tell her sisters the direction of the food
in relation to the sun. When inside the hive, the dancer cannot use the sun, so she
uses gravity instead. The direction of the sun is represented by the top of the hive
wall. If she runs straight up, this means that the feeding place is in the same
direction as the sun. However, if, for example, the feeding place is 40º to the left of
the sun, then the dancer would run 40º to the left of the vertical line. This was to be
the first of von Frisch’s remarkable discoveries. Soon he would also discover a
number of other remarkable facts about how bees communicate and, in doing so,
revolutionise the study of animal behaviour generally.'''

if isinstance(text, str):
    text = text.lower()

    punctuation = [',', '.', ';', ':', '*', '!', '?', " ", "\n"]
    text_new = []
    text_2 = ''.join(i for i in text if i not in punctuation)

    for i in text_2:
        if i.isalpha():
            text_new.append(i)

        print(text_new)
else:
    return None

tokenize(en_text)

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """


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


def print_report(detections: list[list[str | float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
