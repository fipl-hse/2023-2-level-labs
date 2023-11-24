"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator,
                                           NGramLanguageModel, TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    corpus = TextProcessor('_')
    encoded_corpus = corpus.encode(text)
    if encoded_corpus:
        result = corpus.decode(encoded_corpus)
    model = NGramLanguageModel(encoded_corpus, 7)
    greedy_text_generator = GreedyTextGenerator(model, processor)
    greedy_text_generator.run(51, 'Vernon')
    assert result


if __name__ == "__main__":
    main()
