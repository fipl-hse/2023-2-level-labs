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
        text_processor = main_py.TextProcessor('_')
        encoded_text = text_processor.encode(text)
        print(encoded_text)
        print(text_processor.decode(encoded_text))

        model = main_py.NGramLanguageModel(encoded_text, 7)
        greedy_text_generator = main_py.GreedyTextGenerator(model, text_processor)
        generated_text = greedy_text_generator.run(51, 'Vernon')

    result = generated_text
    assert result


if __name__ == "__main__":
    main()
