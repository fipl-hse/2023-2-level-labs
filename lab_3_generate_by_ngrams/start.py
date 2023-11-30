"""
Generation by NGrams starter
"""
import lab_3_generate_by_ngrams.main as main_py


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    split_text = main_py.TextProcessor('_')
    encoded_text = split_text.encode(text)
    if not (isinstance(encoded_text, tuple) and encoded_text):
        return
    result = split_text.decode(encoded_text)
    print(result)
    model = main_py.NGramLanguageModel(encoded_text[:100], 3)
    print(model.build())
    lang_model = main_py.NGramLanguageModel(encoded_text, 7)
    greedy_text = main_py.GreedyTextGenerator(lang_model, split_text)
    print(greedy_text.run(51, 'Vernon'))
    greedy = main_py.BeamSearchTextGenerator(lang_model, split_text, 7)
    print(greedy.run('Vernon', 56))

    assert result


if __name__ == "__main__":
    main()
