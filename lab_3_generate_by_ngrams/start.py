"""
Generation by NGrams starter
"""
from main import TextProcessor, NGramLanguageModel


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    decoded = str(processor.decode(encoded))
    result = decoded
    print(result[:100])

    model = NGramLanguageModel(encoded[:100], n_gram_size=3)
    result = model.build()
    print(result)
    assert result

if __name__ == "__main__":
    main()
