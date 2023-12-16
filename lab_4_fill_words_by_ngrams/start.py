"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_4_fill_words_by_ngrams.main import NGramLanguageModel, TopPGenerator, WordProcessor


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor(end_of_word_token='.')
    encoded = word_processor.encode(text)
    if not isinstance(encoded, tuple) or not encoded:
        raise ValueError('Type input is inappropriate or input argument is empty.')
    language_model = NGramLanguageModel(encoded, n_gram_size=2)
    language_model.build()
    generator = TopPGenerator(language_model, word_processor, p_value=0.5)
    generated_text = generator.run(seq_len=51, prompt='Vernon')
    print(generated_text)
    result = generated_text
    assert result


if __name__ == "__main__":
    main()
