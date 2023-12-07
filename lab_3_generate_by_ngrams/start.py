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
        text = text_file.read()[:1000]

    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    if not (isinstance(encoded, tuple) and encoded):
        return

    decoded = str(processor.decode(encoded))

    language_model = NGramLanguageModel(encoded, 7)
    language_model.build()
    generator = GreedyTextGenerator(language_model, processor)

    result = generator.run(51, "Vernon")

    beam_generator = BeamSearchTextGenerator(language_model, processor, 7)

    result = beam_generator.run("Vernon", 56)

    assert result


if __name__ == "__main__":
    main()
