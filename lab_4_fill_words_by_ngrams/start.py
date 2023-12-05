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
    word_proc = WordProcessor('<eow>')
    encoded = word_proc.encode(text)
    lang_model = NGramLanguageModel(encoded, n_gram_size=2)
    lang_model.build()
    top_p = TopPGenerator(language_model=lang_model, word_processor=word_proc, p_value=0.5)
    result = top_p.run(51, "Vernon")
    print(result)

    generators = GeneratorTypes()
    generators_dict = {generators.greedy: GreedyTextGenerator(lang_model, word_proc),
                       generators.beam_search: BeamSearchTextGenerator(lang_model, word_proc, 5),
                       generators.top_p: top_p}
    quality_check = QualityChecker(generators_dict, lang_model, word_proc)
    run_check = quality_check.run(100, 'The')
    result = [str(cur_check) for cur_check in run_check]
    print("\n".join(result))
    assert result


if __name__ == "__main__":
    main()
