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
    text_processor = main_py.TextProcessor(end_of_word_token='_')
    encoded_text = text_processor.encode(text)
    if not isinstance(encoded_text, tuple) and encoded_text:
        return

    result = str(tuple(text_processor.decode(encoded_text)))
    print(result)

    model = main_py.NGramLanguageModel(encoded_text[:100], 3)
    print(model.build())

    model2 = main_py.NGramLanguageModel(encoded_text, 7)
    greedy_text_generator = main_py.GreedyTextGenerator(model2, text_processor)
    print(greedy_text_generator.run(51, 'Vernon'))

    greedy = main_py.BeamSearchTextGenerator(model2, text_processor, 7)
    print(greedy.run('Vernon', 51))
    assert result


if __name__ == "__main__":
    main()
