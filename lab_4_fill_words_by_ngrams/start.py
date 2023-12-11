"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import NGramLanguageModel
from lab_4_fill_words_by_ngrams.main import (TopPGenerator, WordProcessor)

def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None

    word_processor = WordProcessor(end_of_word_token='.')
    encoded_text = word_processor.encode(text)

    lang_model = NGramLanguageModel(encoded_text, 2)
    lang_model.build()

    top_p = TopPGenerator(lang_model, word_processor, 0.5)
    result = top_p.run(51, 'Vernon')
    print(result)
    assert result
if __name__ == "__main__":
    main()
