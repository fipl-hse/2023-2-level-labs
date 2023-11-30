"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
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
    if encoded_corpus and isinstance(encoded_corpus, tuple):
        result = corpus.decode(encoded_corpus)
    model = NGramLanguageModel(encoded_corpus, 7)
    greedy_text_generator = GreedyTextGenerator(model, corpus)
    greedy_text_generator.run(51, 'Vernon')
    beam_search_generator = BeamSearchTextGenerator(model, corpus, 7)
    beam_search_generator.run('Vernon', 56)
    assert result


if __name__ == "__main__":
    main()
