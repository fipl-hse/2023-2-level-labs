def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if not isinstance(text, str):
        return None
    else:
        tokens = (text.lower())
        punctuation = " !#$%&'\'()*+,-./:;<=>?@[\\]^_`{|}~1234567890"
        for token in tokens:
            if token in punctuation:
                tokens = tokens.replace(token, '')
        tokens = list(tokens)
        return tokens

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    freq = {}
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    for token in tokens:
        freq[token] = (1 if token not in freq else freq[token] + 1)
    for token in freq:
        freq[token] /= len(tokens)
    return freq

def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(tokens, list):
        return None
    frequencies = calculate_frequencies(tokenize(text))
    if isinstance(frequencies, dict):
        return {'name': language, 'freq': frequencies}
    return None