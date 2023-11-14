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
    text_processor = TextProcessor(end_of_word_token='_')
    encoded = text_processor.encode(text)
    result = encoded
    if encoded:
        decoded = text_processor.decode(encoded)
        if decoded:
            print(f"Results:\nEncoded text:\n{encoded[:200]}\nDecoded text:\n{decoded[:200]}")
            result = decoded
    assert result


if __name__ == "__main__":
    main()
