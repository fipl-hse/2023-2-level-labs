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
    text_processor = TextProcessor(end_of_word_token='_')
    encoded = text_processor.encode(text)
    if isinstance(encoded, tuple) and encoded:
        decoded = str(text_processor.decode(encoded))
        result = decoded
        print('Text:', text[:100], '\nDecoded:', decoded[:100], sep='\n', end='\n\n')

        model_6 = NGramLanguageModel(encoded[:1500], 3)
        print(model_6.build())
        greedy_text_generator = GreedyTextGenerator(model_6, text_processor)
        print(greedy_text_generator.run(51, 'Vernon'))

        beam_search_generator = BeamSearchTextGenerator(model_6, text_processor, 7)
        print(beam_search_generator.run('Vernon', 56))
        assert result


if __name__ == "__main__":
    main()
