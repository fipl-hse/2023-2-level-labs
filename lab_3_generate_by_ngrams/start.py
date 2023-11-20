"""
Generation by NGrams starter
"""
from main import NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read().replace('\n', ' ')
    processor = TextProcessor('_')
    encoded_text = processor.encode(text)
    print('Encoding result:', encoded_text)
    print('Decoding result:', processor.decode(encoded_text))
    language_model = NGramLanguageModel(encoded_text, 7)
    language_model.build()
    result = None
    assert result


if __name__ == "__main__":
    main()
