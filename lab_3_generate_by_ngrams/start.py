"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator, \
    NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor('_')
    encoded = processor.encode(text)

    if not encoded:
        return None

    result = processor.decode(encoded)

    if not result:
        return None

    print(result[:100])

    model_for_build = NGramLanguageModel(encoded[:10], 2)
    print(model_for_build.build())

    model = NGramLanguageModel(encoded, 7)
    greedy_text_generator = GreedyTextGenerator(model, processor)
    print(greedy_text_generator.run(51, 'Vernon'))

    beam_search_generator = BeamSearchTextGenerator(model, processor, 7)
    print(beam_search_generator.run('Vernon', 56))

    assert result


if __name__ == "__main__":
    main()
