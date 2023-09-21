def tokenize(text: str) -> list[str] | None:
    text = text.lower()
    tokens = list()
    for el in text:
        if el.isalpha():
            tokens.append(el)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    frequency = {}
    for el in tokens:
        if el not in frequency:
            frequency[el] = 1
        else:
            frequency[el] += 1

    return frequency


calculate_frequencies(tokenize('Hey! How are you?'))
