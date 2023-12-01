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
    if not (isinstance(encoded_corpus, tuple) or encoded_corpus):
        return
    decoded_text = text_processor.decode(encoded_corpus)
    language_model = NGramLanguageModel(encoded_corpus, 7)
    language_model.build()
    generator = GreedyTextGenerator(language_model, text_processor)
    result = generator.run(51, "Vernon")
    assert result


if __name__ == "__main__":
    main()
