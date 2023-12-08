"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import (GreedyTextGenerator,
                                           NGramLanguageModel, TextProcessor)

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_processed = TextProcessor('_')
#     encoded_text = text_processed.encode('''Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
# that they were perfectly normal, thank you very much.''')
#     print(encoded_text)
#     decoded_text = text_processed.decode(encoded_text)
#     print(decoded_text)
#     result = decoded_text
    encoded_text = text_processed.encode(text)
    decoded_text = text_processed.decode(encoded_text)
    print(encoded_text[:40], decoded_text[:40], sep='\n')
    language_model = NGramLanguageModel(encoded_text, 7)
    # freqs = n_grams.build()
    greedy_gen = GreedyTextGenerator(language_model, text_processed)
    generate = greedy_gen.run(51, 'Vernon')
    result = generate
    print(result)
    assert result


if __name__ == "__main__":
    main()
