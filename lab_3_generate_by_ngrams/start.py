"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (BeamSearcher, GreedyTextGenerator,
                                           NGramLanguageModel, TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    split_text = TextProcessor('_')
    encoded_text = split_text.encode(text)
    if encoded_text:
        n_grams = NGramLanguageModel(encoded_text, 7)
        print(n_grams.build)
    greedy_text = GreedyTextGenerator(n_grams, split_text)
    result = greedy_text.run(51, 'Vernon')
    print(result)
    assert result

if __name__ == "__main__":
    main()
