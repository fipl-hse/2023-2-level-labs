"""
BPE Tokenizer starter
"""
from lab_2_tokenize_by_bpe.main import calculate_bleu, decode, encode, load_vocabulary


def main() -> None:
    """
    Launches an implementation
    """
    with open('./assets/for_translation_ru_raw.txt', encoding='utf-8') as prediction_file:
        text = prediction_file.read()
    vocabulary = load_vocabulary('./assets/vocab.json')
    encoded_text = encode(text, vocabulary, '\u2581', None, '<unk>')
    with open('./assets/for_translation_en_encoded.txt', encoding='utf-8') as response_file:
        model_response_str = response_file.read().split()
    model_response = [int(token) for token in model_response_str]
    decoded_response = decode(model_response, vocabulary, None)
    with open('./assets/for_translation_en_raw.txt', encoding='utf-8') as ideal_file:
        ideal_translation = ideal_file.read()
    if decoded_response:
        result = calculate_bleu(decoded_response.replace('\u2581', ' '), ideal_translation)
    else:
        result = None
    assert result, "Encoding is not working"
    print(result)


if __name__ == "__main__":
    main()
