def tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None
    text = text.lower()
    new_text = ""
    for symbol in text:
        if symbol.isalpha():
            new_text += symbol
    tokens = list(new_text)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    all_tokens = len(tokens)
    calc = {}.fromkeys(tokens, 0)
    for token in tokens:
        calc[token] += 1
    for key in calc:
        calc[key] /= all_tokens
    return calc


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    lang_prof = {"name": language, "freq": freq_dict}
    return lang_prof


def calculate_mse(predicted: list, actual: list) -> float | None:
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    p = len(predicted)
    difference_square = []
    for i in range(p):
        difference_square.append((actual[i] - predicted[i]) ** 2)
    mse = sum(difference_square) / p
    return mse


def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if not ('name' and 'freq') in unknown_profile or not ('name' and 'freq') in profile_to_compare:
        return None
    unknown_freq = list(unknown_profile['freq'].values())
    to_compare_freq = list(profile_to_compare['freq'].values())
    mse_to_compare = calculate_mse(unknown_freq, to_compare_freq)
    return mse_to_compare

