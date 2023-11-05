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
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    if isinstance(encoded, tuple) and encoded:
        decoded = str(processor.decode(encoded))
        print('Text:', text[:100], '\nDecoded:', decoded[:100], sep='\n', end='\n\n')
        result = decoded

        ng_model = NGramLanguageModel(encoded[:100], n_gram_size=3)
        print('Started building n-grams!')
        result = f'Successful build? {ng_model.build()} Yeah :)'
        print(result)

        model_6 = NGramLanguageModel(encoded, 7)
        greedy_text_generator = GreedyTextGenerator(model_6, processor)
        print(greedy_text_generator.run(51, 'Vernon'))
        
        assert result



if __name__ == "__main__":
    main()
