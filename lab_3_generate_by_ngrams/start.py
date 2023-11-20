"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
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
    result = split_text.decode(encoded_text)
    model = NGramLanguageModel(encoded_text[:100], 3)
    print(model.build())
    lang_model = NGramLanguageModel(encoded_text, 7)
    greedy_text = GreedyTextGenerator(lang_model, split_text)
    print(greedy_text.run(51, 'Vernon'))
    greedy = BeamSearchTextGenerator(lang_model, split_text, 7)
    print(greedy.run('Vernon', 56))

    assert result

if __name__ == "__main__":
    main()
