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
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    if not(isinstance(encoded, tuple) and encoded):
        return

    decoded = str(processor.decode(encoded))
    result = decoded

    n_gram_model = NGramLanguageModel(encoded[:100], n_gram_size=3)
    model_7 = NGramLanguageModel(encoded, 7)
    greedy_text_generator = GreedyTextGenerator(model_7, processor)
    print(greedy_text_generator.run(51, 'Vernon'))

    beam_search_generator = BeamSearchTextGenerator(model_7, processor, 7)
    print(beam_search_generator.run('Vernon', 56))
    assert result

    decoded_story = str(story.decode(encoded))
    result = decoded_story
    print(result[:100])

    lang_model = NGramLanguageModel(encoded[:50], 3)
    print("checks if the building is successful: ", lang_model.build())

    model2 = NGramLanguageModel(encoded, 7)
    greedy = GreedyTextGenerator(model2, story)
    print(greedy.run(51, "Vernon"))

    beam_search = BeamSearchTextGenerator(model2, story, 7)
    print(beam_search.run('Vernon', 56))

    assert result

if __name__ == "__main__":
    main()
