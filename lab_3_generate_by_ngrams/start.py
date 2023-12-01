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

    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    if isinstance(encoded, tuple) and encoded:
        decoded = str(processor.decode(encoded))
        result = decoded

        # ng_model = NGramLanguageModel(encoded, n_gram_size=3)

        model = NGramLanguageModel(encoded, 7)
        # greedy_text_generator = GreedyTextGenerator(model_6, processor)

        beam_search_generator = BeamSearchTextGenerator(model, processor, 7)
        print(beam_search_generator.run('Vernon', 56))
        # result = beam_search_generator.run('Vernon', 56)
        assert result


if __name__ == "__main__":
    main()
