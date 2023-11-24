"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_processor = TextProcessor('_')
    encoded = text_processor.encode(text)
    decoded = text_processor.decode(encoded)
    result = decoded
    print(result)

    n_gram_language_model = NGramLanguageModel(encoded[:100], 7)
    print(n_gram_language_model.build())
    greedy_text_generator = GreedyTextGenerator(n_gram_language_model, text_processor)
    generated_text = greedy_text_generator.run(51, 'Vernon')
    result = generated_text

    print(generated_text)
    assert result


if __name__ == "__main__":
    main()
