"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel)

from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, QualityChecker,
                                             TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = WordProcessor('<eos>')
    encoded_corpus = processor.encode(text)

    model = NGramLanguageModel(encoded_corpus, 2)
    model.build()

    generator = TopPGenerator(model, processor, 0.5)

    result = generator.run(51, "Vernon")

    gen_types = GeneratorTypes()
    gen_dict = {gen_types.greedy: GreedyTextGenerator(model, processor),
                gen_types.top_p: generator,
                gen_types.beam_search: BeamSearchTextGenerator(model, processor, 5)}

    quality_checker = QualityChecker(gen_dict, model, processor)
    run_check = quality_checker.run(100, 'The')
    for current_check in run_check:
        print(current_check)

    assert result



if __name__ == "__main__":
    main()
