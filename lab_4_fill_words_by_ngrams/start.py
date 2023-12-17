"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_4_fill_words_by_ngrams.main import NGramLanguageModel, TopPGenerator, WordProcessor


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as file:
        text = file.read()
    processor = WordProcessor('.')
    encoded = processor.encode(text)
    if not isinstance(encoded, tuple) or not encoded:
        raise ValueError

    model = NGramLanguageModel(encoded[:10000], 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)
    generated_text = generator.run(51, 'Vernon')
    print(generated_text)
    result = generated_text
    assert result


if __name__ == "__main__":
    main()
