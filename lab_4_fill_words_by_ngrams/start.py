"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
import lab_4_fill_words_by_ngrams.main as main_py

def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None
    word_processor = main_py.WordProcessor('<eow>')
    top_p_generator = main_py.TopPGenerator(language_model, word_processor, 0.5)
    assert word_processor


if __name__ == "__main__":
    main()
