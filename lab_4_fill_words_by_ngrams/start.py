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
    processor = WordProcessor('_')
    encoded = processor.encode(text)
    model = NGramLanguageModel(encoded, 2)
    model.build()
    p_generator = TopPGenerator(model, processor, 0.5)
    print(p_generator.run(51, 'Vernon'), '\n')

    type_storage = GeneratorTypes()
    generators = {
        type_storage.greedy: GreedyTextGenerator(model, processor),
        type_storage.top_p: p_generator,
        type_storage.beam_search: BeamSearchTextGenerator(model, processor, 5)
    }
    checker = QualityChecker(generators, model, processor)
    result = checker.run(100, 'The')
    for report in result:
        print(report)
    assert result


if __name__ == "__main__":
    main()
