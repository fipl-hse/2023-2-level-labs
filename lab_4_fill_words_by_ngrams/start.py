"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, NGramLanguageModel, QualityChecker,
                                             TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None
    processor = WordProcessor('<eow>')
    text_encoded = processor.encode(text)
    model = NGramLanguageModel(text_encoded, 2)
    model.build()
    gen_top_p = TopPGenerator(model, processor, 0.5)
    generation_1 = gen_top_p.run(51, 'Vernon')
    result = generation_1
    generator_types = GeneratorTypes()
    generators = {generator_types.greedy: GreedyTextGenerator(model, processor),
                  generator_types.top_p: gen_top_p,
                  generator_types.beam_search: BeamSearchTextGenerator(model, processor, 5)}
    # quality_checker = QualityChecker(generators, model, processor)
    # results = quality_checker.run(100, 'The')
    # for res in results:
    #     print(res)
    assert result


if __name__ == "__main__":
    main()
