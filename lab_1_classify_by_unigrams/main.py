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


create_language_profile('en', 'jjhdf hkjddshu gdfd')
