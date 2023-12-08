"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from main import NGramLanguageModel, TopPGenerator, WordProcessor

def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = WordProcessor('<eow>')
    text_encoded = processor.encode(text)
    model = NGramLanguageModel(text_encoded, 2)
    model.build()
    gen_topP = TopPGenerator(model, processor, 0.5)
    generation_1 = gen_topP.run(51, 'Vernon')
    print(generation_1)
    result = generation_1
    assert result


if __name__ == "__main__":
    main()
