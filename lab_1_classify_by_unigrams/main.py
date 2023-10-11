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
    if not isinstance(text,str):
        return None
    text=text.lower()
    wronglist=(list(text))
    goodlist=[]
    for item in wronglist:
        if item.isalpha():
            goodlist.append(item)
    return(goodlist)


tokenize("texugyTTRETHFDFG546//t")



def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens,list):
        return None
    numberallitems=len(tokens)
    freq_dict={}
    for item in tokens:
        numberi=tokens.count(item)
        freqi=numberi/numberallitems
        freq_dict[item]=freqi
    return(freq_dict)
calculate_frequencies(['t', 'e', 'x', 'u', 'g', 'y', 't', 't', 'r', 'e', 't', 'h', 'f', 'd', 'f', 'g', 't'])


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str):
        return None
    if not isinstance(text, str):
        return None
    goodlist=tokenize(text)
    freq_dict=calculate_frequencies(goodlist)
    profilel={"name": language, "freq": {}}

    for key, value in freq_dict.items():
        profilel["freq"][(key)]=(value)
    return(profilel)
create_language_profile("en", "yft45dytdTU")


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isistance(predicted,list):
        return None
    if not isinstance(actual,list):
        return None
    if len(predicted) != len(actual):
        return None
    if len(actual)==0:
        return None
    else:
        dlina=len(actual)
        for index in range(dlina):
            numeratory += actual[index]
            numeratorp += predicted[index]
            numerator += (numeratory - numeratorp)**2
    mse=(numerator/dlina)
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
