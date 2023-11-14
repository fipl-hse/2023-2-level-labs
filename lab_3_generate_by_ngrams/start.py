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
    textprocessor = TextProcessor(end_of_word_token='_')
    encoded_text = textprocessor.encode(text)
    decoded_text = textprocessor.decode(encoded_text)
    ngramlanguagemodel = NGramLanguageModel(encoded_text, n_gram_size=7)
    greedytextgenerator = GreedyTextGenerator(ngramlanguagemodel, textprocessor)
    predicted_text = greedytextgenerator.run(51, 'Vernon')
    result = predicted_text
    print(result)
    assert result



if __name__ == "__main__":
    main()
