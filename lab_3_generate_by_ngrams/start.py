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
    encoded_corpus = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_corpus)
    language_model = main_py.NGramLanguageModel(encoded_corpus, 7)
    print(language_model.build())
    greedy_generator = main_py.GreedyTextGenerator(language_model, text_processor)
    generated_text = greedy_generator.run(51, 'Vernon')
    print(generated_text)
    beam_search_generator = main_py.BeamSearchTextGenerator(language_model, text_processor, 7)
    print(beam_search_generator.run('Vernon', 56))
    result = decoded_text
    assert result


if __name__ == "__main__":
    main()
