"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             NGramLanguageModel, QualityChecker, TopPGenerator,
                                             WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = WordProcessor('<eow>')
    encoded = processor.encode(text)
    model = NGramLanguageModel(encoded, 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)
    result = generator.run(51, 'Vernon')
    print(result, '\n')
    assert result


if __name__ == "__main__":
    main()
