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
    if not isinstance(encoded, tuple) or encoded is None:
        return
    language_model = NGramLanguageModel(encoded, 2)
    language_model.build()
    generator = TopPGenerator(language_model, processor, 0.5)
    generated_text = generator.run(51, 'Vernon')
    print(generated_text)
    result = generated_text
    assert result


if __name__ == "__main__":
    main()
