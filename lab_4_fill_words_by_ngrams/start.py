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
    word_proc = WordProcessor('<eos>')
    encoded_text = word_proc.encode(text)
    model = NGramLanguageModel(encoded_text, 2)
    model.build()
    generated = TopPGenerator(model, word_proc, 0.5)
    result = generated.run(51, 'Vernon')
    print(result)
    assert result


if __name__ == "__main__":
    main()
