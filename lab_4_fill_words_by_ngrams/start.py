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

    processor = WordProcessor('<eos>')
    encoded_corpus = processor.encode(text)

    model = NGramLanguageModel(encoded_corpus, 2)
    model.build()

    generator = TopPGenerator(model, processor, 0.5)

    result = generator.run(51, "Vernon")

    assert result


if __name__ == "__main__":
    main()
