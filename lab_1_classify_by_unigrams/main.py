def tokenize(text: str) -> list[str] | None:
    pass
    if not isinstance(text, str):
        return None
    tokens = []
    for char in text.lower():
        if char.isalpha():
            tokens.append(char)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None

    token_count = len(tokens)
    frequency_dict = {}

    for token in tokens:
        if token in frequency_dict:
            frequency_dict[token] += 1
        else:
            frequency_dict[token] = 1

    for token, count in frequency_dict.items():
        frequency_dict[token] = count / token_count

    return frequency_dict


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if not isinstance(language, str) or not isinstance(text, str):
        return None

    tokens = tokenize(text)
    frequencies = calculate_frequencies(tokens)

    profile = {'name': language, 'freq': frequencies}
    return profile
