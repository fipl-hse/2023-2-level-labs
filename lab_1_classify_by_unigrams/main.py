"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str] | None:
    text = text.lower()
    text = text.replace(' ', '')
    text = text.replace('\n', 'replace')
    nopunc_str = '''!()-[]{};:/?@]#$%^'",.&*_~'''
    for punc in text:
        if punc in nopunc_str:
            text = text.replace(punc, "")
        if punc.isdigit():
            text = text.replace(punc, "")
    token = list(text)
    return token
