"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel, TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    textprocessor = TextProcessor('_')
    encoded_text = textprocessor.encode(text)
    decoded_text = textprocessor.decode(encoded_text)
    ngramlanguagemodel = NGramLanguageModel(encoded_text[:100], 3)
    result = ngramlanguagemodel.build()
    print(result)
    ngramlanguagemodel_2 = NGramLanguageModel(encoded_text, 7)
    greedytextgenerator = GreedyTextGenerator(ngramlanguagemodel_2, textprocessor)
    predicted_text = greedytextgenerator.run(51, 'Vernon')
    print(predicted_text)
    beamsearchtextgenerator = BeamSearchTextGenerator(ngramlanguagemodel_2, textprocessor, 7)
    predicted_text_2 = beamsearchtextgenerator.run('Vernon', 56)
    print(predicted_text_2)
    assert result



if __name__ == "__main__":
    main()
