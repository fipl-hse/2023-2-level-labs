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
    result = None
    text_processed = TextProcessor('_')
    encoded_text = text_processed.encode(text)
    print(encoded_text)
    decoded_text = text_processed.decode(encoded_text)
    print(decoded_text)

if __name__ == "__main__":
    main()
