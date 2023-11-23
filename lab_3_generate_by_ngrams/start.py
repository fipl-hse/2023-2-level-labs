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
    if isinstance(encoded_text, tuple) and encoded_text:
        decoded_text = text_processor.decode(encoded_text)
        print(decoded_text)

        language_model = main_py.NGramLanguageModel(encoded_text[:100], 3)
        n_grams = language_model.build()
        print(n_grams)

        lang_model2 = main_py.NGramLanguageModel(encoded_text, 7)
        greedy_generator = main_py.GreedyTextGenerator(lang_model2, text_processor)
        generated_text = greedy_generator.run(51, 'Vernon')
        print(generated_text)

        beam_search_generator = main_py.BeamSearchTextGenerator(lang_model2, text_processor, 7)
        resulted_text = beam_search_generator.run('Vernon', 56)
        print(resulted_text)

        result = decoded_text
        assert result


if __name__ == "__main__":
    main()
