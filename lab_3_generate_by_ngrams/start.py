"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_processor = TextProcessor(end_of_word_token = '_')
    encoded_corpus = text_processor.encode(text)
    if isinstance(encoded_corpus, tuple) and encoded_corpus:
        decoded_text = str(text_processor.decode(encoded_corpus))
        print('Text:', text[:100], '\nDecoded:', decoded_text[:100], sep='\n', end='\n')
        result = decoded_text
    assert result


if __name__ == "__main__":
    main()

