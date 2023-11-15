"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor



def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    split_text = TextProcessor('_')
    encoded_text = split_text.encode(text)
    if not encoded_text:
        return None
    n_grams = NGramLanguageModel(encoded_text, 7)
    greedy_text = GreedyTextGenerator(n_grams, split_text)
    result = greedy_text.run(51, 'Vernon')
    print(result)


    #result = split_text.decode(encoding)
    #print('Encoding of the text:', encoding)
    #print('Decoding of the text:', result)
    assert result

if __name__ == "__main__":
    main()
