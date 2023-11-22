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
    text_processor = TextProcessor('_')
    encoded_corpus = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_corpus)
    result = decoded_text
    print(result)
    assert result



if __name__ == "__main__":
    main()
