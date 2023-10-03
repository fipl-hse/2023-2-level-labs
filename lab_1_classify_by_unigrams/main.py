def tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None
    text = text.lower()
    tokens = list()
    for el in text:
        if el.isalpha():
            tokens.append(el)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if not isinstance(tokens, list):
        return None
    for text in tokens:
        if not isinstance(text, str):
            return None
    frequency = {}
    symbols = len(tokens)
    for el in tokens:
        if el not in frequency:
            frequency[el] = 1
        else:
            frequency[el] += 1
    for k, v in frequency.items():
        frequency[k] = v / symbols
    return frequency


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if not (isinstance(language, str) and isinstance(text, str)):
        return None
    freq = calculate_frequencies(tokenize(text))
    profile = {'name': language,
               'freq': freq}
    return profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    if not len(predicted) == len(actual):
        return None
    if not (isinstance(predicted, list) and isinstance(actual, list)):
        return None
    difference = []
    total = 0
    for el1 in predicted:
        for el2 in actual:
            difference += (el2 - el1) ** 2
    for dif in difference:
        total += dif
    mse = total / len(actual)
    return mse


def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    if not (isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict)):
        return None
    if ('name' and 'freq') not in unknown_profile and ('name' and 'freq') not in profile_to_compare:
        return None
    keys = sorted(set(unknown_profile.keys()) | set(profile_to_compare.keys()))
    compare_un = []
    compare_pr = []
    for key in keys:
        compare_un.append(unknown_profile.get(key, 0))
        compare_pr.append(profile_to_compare.get(key, 0))
    mse_calc = calculate_mse(compare_un, compare_pr)
    return mse_calc

