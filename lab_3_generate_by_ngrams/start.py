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
    result = None


    encoded_text = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_text)
    print(decoded_text)

    language_model = main_py.NGramLanguageModel(encoded_text[:100], 7)
    print(language_model.build())

    greedy_generator = main_py.GreedyTextGenerator(language_model, text_processor)
    generated_text = greedy_generator.run(51, 'Vernon')
    result = generated_text
    print(generated_text)

     assert result



if __name__ == "__main__":
    main()
