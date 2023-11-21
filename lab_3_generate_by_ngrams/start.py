"""
Generation by NGrams starter
"""
import lab_3_generate_by_ngrams.main as main_py
#from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel, GreedyTextGenerator

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None
    processor = main_py.TextProcessor('_')
    encoded_text = processor.encode(text)
    result = processor.decode(encoded_text)
    print(encoded_text)
    print(result)

    lang_model = main_py.NGramLanguageModel(encoded_text, 2)
    build = lang_model.build()
    print(build)

    language_model = main_py.NGramLanguageModel(encoded_text, 7)
    greedy_gen = main_py.GreedyTextGenerator(language_model, processor)
    generated_text = greedy_gen.run(51, 'Vernon')
    print(generated_text)
    assert result


if __name__ == "__main__":
    main()
