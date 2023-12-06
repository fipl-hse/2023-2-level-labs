"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_4_fill_words_by_ngrams.main import (WordProcessor, NGramLanguageModel, TopPGenerator)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None

    word_processor = WordProcessor('<eow>')
    encoded_text = word_processor.encode(text)
    language_model = NGramLanguageModel(encoded_text, 2)
    language_model.build()
    top_p_generator = TopPGenerator(language_model, word_processor, 0.5)
    result = top_p_generator.run(51, 'Vernon')
    print(result)

    assert result


if __name__ == "__main__":
    main()
