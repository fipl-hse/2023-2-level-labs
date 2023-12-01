"""
Lab 1
Language detection
"""
import string


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    text = text.lower()
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    for punc in text:
        if punc in string.punctuation or punc.isdigit():
            text = text.replace(punc, "")
    return list(text)

