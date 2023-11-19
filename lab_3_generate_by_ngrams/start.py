"""
Generation by NGrams starter
"""
from main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    result = processor.decode(encoded)
    print(result[:100])

    model = NGramLanguageModel(encoded[:100], n_gram_size=3)
    print(model.build())

    model2 = NGramLanguageModel(encoded, 7)
    greedy_text_generator = GreedyTextGenerator(model2, processor)
    print(greedy_text_generator.run(51, 'Vernon'))

    assert result


if __name__ == "__main__":
    main()
