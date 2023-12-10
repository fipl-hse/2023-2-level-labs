"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import

from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, BeamSearchTextGenerator,
                                             NGramLanguageModel, TopPGenerator,
                                             QualityChecker, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor("<eos>")
    encoded_text = word_processor.encode(text)
    lang_model = NGramLanguageModel(encoded_text, 2)
    lang_model.build()

    top_p_generator = TopPGenerator(lang_model, word_processor, 0.5)
    result = top_p_generator.run(51, "Vernon")
    print(result)

    assert result


if __name__ == "__main__":
    main()
