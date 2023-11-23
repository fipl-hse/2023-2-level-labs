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
    text_processor = TextProcessor('_')
    encoded = text_processor.encode(text)
    if not isinstance(encoded, tuple) and not encoded:
        return
    result = text_processor.decode(encoded)[:100]
    print(result)
    n_gram_model = NGramLanguageModel(encoded[:100], n_gram_size=3)
    freq = n_gram_model.build()
    if not freq:
        return
    result = freq
    print(result)
    other_n_gram_model = NGramLanguageModel(encoded, n_gram_size=7)
    greedy_generator = GreedyTextGenerator(other_n_gram_model, text_processor)
    result = greedy_generator.run(51, "Vernon")
    print(result)

    assert result


if __name__ == "__main__":
    main()
