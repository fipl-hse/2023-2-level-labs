"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import NGramLanguageModel, TextProcessor, GreedyTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    # result = None
    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_text)
    language_model = NGramLanguageModel(encoded_text, 7)
    # freqs = n_grams.build()
    greedy_gen = GreedyTextGenerator(language_model, text_processor)
    generate = greedy_gen.run(51, 'Vernon')
    result = generate
    print(result)
    assert result


if __name__ == "__main__":
    main()
