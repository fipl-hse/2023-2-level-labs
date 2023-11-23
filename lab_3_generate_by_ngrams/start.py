"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel, GreedyTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)

    if encoded_text:
        result = text_processor.decode(encoded_text)
        print(result)

        language_model = NGramLanguageModel(encoded_text[:10], 2)
        print(language_model.build())

        model = NGramLanguageModel(encoded_text, 7)
        greedy_generator = GreedyTextGenerator(model, text_processor)
        print(greedy_generator.run(51, 'Vernon'))

        assert result


if __name__ == "__main__":
    main()
