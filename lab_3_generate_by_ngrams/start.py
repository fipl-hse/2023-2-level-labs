"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel

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
    result = processor.decode(encoded_text)
    print(encoded_text)
    print(result)

    lang_model = NGramLanguageModel(encoded_text, 2)
    build = lang_model.build()
    print(build)

    assert result


if __name__ == "__main__":
    main()
