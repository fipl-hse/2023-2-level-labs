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

    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    result = processor.decode(encoded)

    extracted_n_grams = NGramLanguageModel(encoded_corpus=encoded[:100], n_gram_size=3)
    built_model = extracted_n_grams.build()

    assert result


if __name__ == "__main__":
    main()
