"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel, GreedyTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    story = TextProcessor("_")
    decoded_story = story.decode(story.encode(text))
    encoded = story.encode(text)
    result = decoded_story
    lang_model = NGramLanguageModel(encoded[:50], 3)
    model = lang_model.build()
    model2 = NGramLanguageModel(encoded, 7)
    greedy = GreedyTextGenerator(model2, story)
    print(greedy.run(51, "Vernon"))
    assert result


if __name__ == "__main__":
    main()
