"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    #result = None
    result = TextProcessor('_')
    print(result._tokenize(text=text))
    print(result.encode(text=text))
    assert result


if __name__ == "__main__":
    main()
