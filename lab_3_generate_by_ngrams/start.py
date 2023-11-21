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
    result = None

    processor = TextProcessor('_')
    encoded_text = processor.encode(text)
    print(processor.decode(encoded_text))
    if encoded_text:
        lang_model = NGramLanguageModel(encoded_text, 2)
        print(lang_model.build())

        language_model = NGramLanguageModel(encoded_text, 7)
        greedy_gen = GreedyTextGenerator(language_model, processor)
        generated_text = greedy_gen.run(51, 'Vernon')
        print(generated_text)
    assert result


if __name__ == "__main__":
    main()
