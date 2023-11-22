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
    text_processor = TextProcessor(end_of_word_token='_')
    encoded = text_processor.encode(text)
    if encoded is None or not isinstance(encoded, tuple):
        return
    decoded = text_processor.decode(encoded)
    if decoded is None:
        return
    print(f"{encoded[:100]}\n{decoded[:100]}")
    n_gram_model = NGramLanguageModel(encoded_corpus=encoded[:100], n_gram_size=3)
    freqs = n_gram_model.build()
    if freqs is None:
        return
    result = freqs
    print(result)
    another_model = NGramLanguageModel(encoded_corpus=encoded, n_gram_size=7)
    greedy_generator = GreedyTextGenerator(another_model, text_processor)
    result = greedy_generator.run(51, "Vernon")
    print(result)
    assert result


if __name__ == "__main__":
    main()
