"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, NGramLanguageModel,
                                             TopPGenerator, QualityChecker, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = WordProcessor('.')
    encoded = processor.encode(text)

    if not(isinstance(encoded, tuple) and encoded):
        return None

    model = NGramLanguageModel(encoded[:10000], 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)

    generated = generator.run(51, 'Vernon')
    print(generated)
    result = generated

    generators = GeneratorTypes()
    gen_dict = {generators.greedy:  GreedyTextGenerator(
        model, processor), generators.top_p: generator, generators.beam_search:
        BeamSearchTextGenerator(model, processor, 5)}

    quality = QualityChecker(gen_dict, model, processor)
    checking = quality.run(100, 'The')

    for check in checking:
        result = str(check)
        print(result)

    assert result
    return None


if __name__ == "__main__":
    main()
