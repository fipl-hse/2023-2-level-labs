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
    textprocessor = TextProcessor('_')
    encoded_text = textprocessor.encode(text)
    if encoded_text:
        result = textprocessor.decode(encoded_text)
        print(result)
        model_for_build = NGramLanguageModel(encoded_text[:10], 2)
        print(model_for_build.build())
        model = NGramLanguageModel(encoded_text, 7)
        greedy_text_generator = GreedyTextGenerator(model, textprocessor)
        print(greedy_text_generator.run(51, 'Vernon'))
        beam_search_generator = BeamSearchTextGenerator(model, textprocessor, 7)
        print(beam_search_generator.run('Vernon', 56))
        assert result
    assert isinstance(encoded_text, tuple)
    assert result


if __name__ == "__main__":
    main()
