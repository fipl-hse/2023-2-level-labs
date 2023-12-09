"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel
from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, TopPGenerator,
                                             QualityChecker, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor('<eow>')
    encoded_text = word_processor.encode(text)
    model = NGramLanguageModel(encoded_text, 2)
    model.build()
    top_p = TopPGenerator(model, word_processor, 0.5)
    top_p_result = top_p.run(51, 'Vernon')
    print(top_p_result)
    generator_types = GeneratorTypes()
    generators = {generator_types.top_p: TopPGenerator(model, word_processor, 0.5),
                  generator_types.beam_search: BeamSearchTextGenerator(model, word_processor, 5)}
    quality_check = QualityChecker(generators, model, word_processor)
    result = quality_check.run(100, 'The')
    assert result


if __name__ == "__main__":
    main()
