"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
import lab_4_fill_words_by_ngrams.main as main_py


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None

    word_processor = main_py.WordProcessor('<eow>')
    encoded_text = word_processor.encode(text)
    language_model = main_py.NGramLanguageModel(encoded_text, 2)
    language_model.build()
    top_p_generator = main_py.TopPGenerator(language_model, word_processor, 0.5)
    result = top_p_generator.run(51, 'Vernon')
    print(result)

    generator_types = main_py.GeneratorTypes()
    generators = {generator_types.top_p: main_py.TopPGenerator(language_model, word_processor, 0.5),
                  generator_types.beam_search: main_py.BeamSearchTextGenerator(language_model, word_processor, 5)}
    quality_check = main_py.QualityChecker(generators, language_model, word_processor)
    result = quality_check.run(100, 'The')
    print(result)
    assert result


if __name__ == "__main__":
    main()
