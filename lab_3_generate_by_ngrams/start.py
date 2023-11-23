"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_processor = TextProcessor('_')
    encoded_corpus = text_processor.encode(text)

    if isinstance(encoded_corpus, tuple) and encoded_corpus:
        decoded_text = str(text_processor.decode(encoded_corpus))
        result = decoded_text

        language_model = NGramLanguageModel(encoded_corpus, n_gram_size=3)
        print(language_model.build())

        model_6 = NGramLanguageModel(encoded_corpus, 7)
        greedy_text_generator = GreedyTextGenerator(model_6, text_processor)
        print(greedy_text_generator.run(51, 'Vernon'))

        assert result


if __name__ == "__main__":
    main()
