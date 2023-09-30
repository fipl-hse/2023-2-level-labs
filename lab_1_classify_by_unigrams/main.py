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


calculate_frequencies(tokenize("Hey! How are you?"))


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    lang_prof = {"name": language, "freq": freq_dict}
    return lang_prof


# create_language_profile('eng', "Hey! How are you?")


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
    print(mse)


calculate_mse([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24], [37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23])
