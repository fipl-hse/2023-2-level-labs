"""
Lab 1
Language detection
"""
import json


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
    tokens = []
    for token in text:
        if token.isalpha():
            tokens.append(token)
    return tokens





def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    if not isinstance(tokens, list):
        return None
    frequency_dict = {}
    tokens_number = len(tokens)
    for token in tokens:
        if not isinstance(token, str):
            return None
        token_frequency = tokens.count(token)
        frequency_dict[token] = token_frequency / tokens_number
    return frequency_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    frequencies = calculate_frequencies(tokenize(text))
    if isinstance(frequencies, dict):
        return {'name': language, 'freq': frequencies}
    return None



def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if len(predicted) != len(actual):
        return None
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    mse_sum = 0
    for index, element in enumerate(predicted):
        mse_sum += (element - actual[index]) ** 2
    mse = mse_sum / len(predicted)
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
    if ('name' or 'freq') not in (unknown_profile or profile_to_compare):
        return None
    all_tokens = []
    unk_prof_freq = []
    prof_comp_freq = []
    for letter in unknown_profile.get('freq'):
        all_tokens.append(letter)
    for letter in profile_to_compare.get('freq'):
        if letter not in all_tokens:
            all_tokens.append(letter)
    for token in all_tokens:
        if token in unknown_profile['freq']:
            unk_prof_freq.append(unknown_profile['freq'].get(token))
        else:
            unk_prof_freq.append(0)
        if token in profile_to_compare['freq']:
            prof_comp_freq.append(profile_to_compare['freq'].get(token))
        else:
            prof_comp_freq.append(0)
    return calculate_mse(unk_prof_freq, prof_comp_freq)



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
    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)
    name_1 = str(profile_1['name'])
    name_2 = str(profile_2['name'])
    if isinstance(mse1, float) and isinstance(mse2, float):
        if mse1 > mse2:
            return name_2
        if mse1 < mse2:
            return name_1
        if mse1 == mse2:
            both_keys = [name_1, name_2]
            both_keys.sort()
            return both_keys[0]
    return None



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
    if isinstance(profile, dict):
        return profile
    return None



def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not isinstance(profile, dict):
        return None
    if 'name' not in profile or 'freq' not in profile or 'n_words' not in profile:
        return None
    big_letters = {}
    for n_gram in profile.get('freq').copy():
        if len(n_gram) != 1:
            profile['freq'].pop(n_gram)
        elif n_gram.isupper():
            big_letters[n_gram.lower()] = profile['freq'].pop(n_gram)
        else:
            continue
    for letter in big_letters:
        if letter in profile['freq']:
            profile['freq'][letter] += big_letters[letter]
        else:
            profile['freq'][letter] = big_letters[letter]
    for letter in profile['freq']:
        profile['freq'][letter] = profile['freq'][letter] / profile['n_words'][0]
    profile.pop('n_words')
    if isinstance(profile, dict):
        return profile
    return None




def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    list_of_profiles = []
    for profile in paths_to_profiles:
        language_profile = load_profile(profile)
        if isinstance(language_profile, dict):
            processed_profile = preprocess_profile(language_profile)
        if isinstance(processed_profile, dict):
            list_of_profiles.append(processed_profile)
    return list_of_profiles




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
    for profile in known_profiles:
        mse_list.append((profile.get('name'), compare_profiles(unknown_profile, profile)))
    mse_list.sort(key=lambda a: (a[1], a[0]))
    return mse_list



def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if isinstance(detections, list):
        for profile in detections:
            language = profile[0]
            score = profile[1]
            print(f'{language}: MSE {score:.5f}')

