"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, TopPGenerator, QualityChecker, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor('<eow>')
    encoded = word_processor.encode(text)
    lang_model = NGramLanguageModel(encoded, n_gram_size=2)
    lang_model.build()
    top_p = TopPGenerator(lang_model, word_processor, 0.5)
    generated_text_6 = top_p.run(51, 'Vernon')
    print(generated_text_6)
    gen_types = GeneratorTypes()
    generators = {gen_types.greedy: GreedyTextGenerator(lang_model, word_processor),
                  gen_types.top_p: top_p,
                  gen_types.beam_search: BeamSearchTextGenerator(lang_model, word_processor, 5)}
    quality_check = QualityChecker(generators, lang_model, word_processor)
    generating = quality_check.run(100, 'The')
    result = [str(current) for current in generating]
    print("\n".join(result))
    assert result


if __name__ == "__main__":
    main()
