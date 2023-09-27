"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str]:
    """
        Splits a text into tokens, converts the tokens into lowercase,
        removes punctuation, digits and other symbols
        :param text: a text
        :return: a list of lower-cased tokens without punctuation
        """
    if type(text) != str:
        return None
    text = text.lower()
    for i in text:
        if not i.isalpha() and i != ' ':
            text = text.replace(i, '')
    for i in text:
        if i == ' ':
            text = text.replace(' ','')
    tokens = list(text)
    return tokens



def calculate_frequencies(tokens : list[str] | None) -> dict[str, float] | None:
    """
       Calculates frequencies of given tokens
       :param tokens: a list of tokens
       :return: a dictionary with frequencies
       """
    if type(tokens) != list:
        return None
    for i in tokens:
        if type(i) != str:
            return None
    dictionary = {}
    length = len(tokens)
    for i in tokens:
        if i in dictionary:
            dictionary[i] += 1/length
        else:
            dictionary[i] = 1/length
    return dictionary






def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
      Creates a language profile
      :param language: a language
      :param text: a text
      :return: a dictionary with two keys – name, freq
      """
    if type(language) != str or type(text) != str:
        return None
    new_dictionary = {}
    text = tokenize(text)
    freq = calculate_frequencies(text)
    new_dictionary[language] = freq
    return new_dictionary

def calculate_mse(predicted: list, actual: list) -> float | None:
    """
       Calculates mean squared error between predicted and actual values
       :param predicted: a list of predicted values
       :param actual: a list of actual values
       :return: the score
       """
    if type(predicted) != list or type(actual) != list or len(predicted) != len(actual):
        return None
    cubs = 0
    n = len(actual)
    for i in range(len(actual)):
        cubs += (actual[i] - predicted[i]) ** 2
    mse = cubs / n
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
    if type(unknown_profile) != dict or type(profile_to_compare) != dict:
        return None
    key1 = 'name'
    key2 = 'freq'
    if key1 not in unknown_profile or key2 not in unknown_profile or key1 not in profile_to_compare or key2 not in profile_to_compare:
        return None
    new_list = []
    list1 = []
    list2 = []
    unknown_freq = unknown_profile['freq']
    compare_freq = profile_to_compare['freq']
    unknown_list = list(unknown_freq.values())
    compare_list = list(compare_freq.values())
    new_list = unknown_list + compare_list
    new = list(set(new_list))
    for i in range(len(new)):
        if new[i] in unknown_freq.keys():
            list1.append(unknown_freq[new[i]])
        else:
            list1.append(float(0))
    for i in range(len(new)):
        if new[i] in compare_freq.keys():
            list2.append(compare_freq[new[i]])
        else:
            list2.append(float(0))
    mse = calculate_mse(list1, list2)
    return mse



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
    if type(unknown_profile) != dict or type(profile_1) != dict or type(profile_2) != dict:
        return None
    if compare_profiles(unknown_profile, profile_1) < compare_profiles(unknown_profile, profile_2):
        return profile_1['name']
    else:
        return profile_2['name']


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
