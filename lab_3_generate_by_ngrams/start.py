"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)

    if encoded_text:
        result = text_processor.decode(encoded_text)
        print(result)
        build_model = NGramLanguageModel(encoded_text[:10], 2)
        print(build_model.build())
        language_model = NGramLanguageModel(encoded_text, 7)
        greedy_text_generator = GreedyTextGenerator(language_model, text_processor)
        print(greedy_text_generator.run(51, 'Vernon'))
        beam_search_generator = BeamSearchTextGenerator(language_model, text_processor, 7)
        print(beam_search_generator.run('Vernon', 56))
        assert result


if __name__ == "__main__":
    main()
