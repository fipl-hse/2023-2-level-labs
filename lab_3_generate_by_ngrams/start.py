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
    story = TextProcessor("_")
    encoded = story.encode(text)
    if not(isinstance(encoded, tuple) and encoded):
        return

    decoded_story = str(story.decode(encoded))
    result = decoded_story

    lang_model = NGramLanguageModel(encoded[:50], 3)
    model2 = NGramLanguageModel(encoded, 7)
    greedy = GreedyTextGenerator(model2, story)
    print(greedy.run(51, "Vernon"))

    beam_search = BeamSearchTextGenerator(model2, story, 7)
    print(beam_search.run('Vernon', 56))
    assert result

if __name__ == "__main__":
    main()
