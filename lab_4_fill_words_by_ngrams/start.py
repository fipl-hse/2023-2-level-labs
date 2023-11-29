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

    processor = WordProcessor('.')
    encoded = processor.encode(text)
    if isinstance(encoded, tuple) and encoded:
        model = NGramLanguageModel(encoded[:10000], 2)
        model.build()
        generator = TopPGenerator(model, processor, 0.5)

        generated = generator.run(51, 'Vernon')
        print(generated)
        result = generated
        assert result


if __name__ == "__main__":
    main()
