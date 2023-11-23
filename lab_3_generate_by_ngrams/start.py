"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel, GreedyTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor('_')
    encoded_text = processor.encode(text)
    decoded_text = processor.decode(encoded_text)
    print(decoded_text[:300])

    lang_model = NGramLanguageModel(encoded_text[:300], 7)
    print(lang_model.build())

    greedy_text_gen = GreedyTextGenerator(lang_model, processor)
    result = greedy_text_gen.run(51, 'Vernon')
    print(result)
    assert result


if __name__ == "__main__":
    main()
