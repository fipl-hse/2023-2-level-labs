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
    text_ = TextProcessor('_')
    result1 = text_.encode(text)
    if result1 is None:
        return
    print(result1[:200])
    result2 = text_.decode(result1)
    if result2 is None:
        return
    print(result2[:200])
    result = result2
    assert result


if __name__ == "__main__":
    main()
