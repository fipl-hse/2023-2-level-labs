import string
def tokenize(text):
    if isinstance(text, str):
        text = text.lower()
        text1 = text.translate(str.maketrans('', '', string.punctuation))
        text2 = text1.replace(" ", "")
        tokens = list(text2)
        return tokens
    else:
        return None

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if isinstance(tokens, list) and all(isinstance(el,str) for el in tokens):
        dictionary = {el: (tokens.count(el) / len(tokens)) for el in tokens}
        return dictionary
    else:
        return None

def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    if isinstance(language, str) and isinstance(text, str):
        tokens = tokenize(text)
        frequency_dict = calculate_frequencies(tokens)
        language_profile = {'name': language, 'freq': frequency_dict}
        return language_profile
    else:
        return None



