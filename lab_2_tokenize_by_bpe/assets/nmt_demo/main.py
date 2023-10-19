"""
Demo module for Neural Machine Translation
"""

from transformers import MarianMTModel, MarianTokenizer  # pylint: disable=import-error


def translate(
        model: MarianMTModel,
        tokenizer: MarianTokenizer,
        input_text: str
) -> str:
    """
    Translation of arbitrary text

    When used with our encode() method, need to pass
    inputs=torch.tensor([encoded_text, 0]) instead of inputs=inputs
    """
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    translated_tokens = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


def load_model(model_name: str) -> tuple[MarianMTModel, MarianTokenizer]:
    """
    Load Hugging Face model by its name
    """
    tokenizer_en = MarianTokenizer.from_pretrained(model_name)
    model_en = MarianMTModel.from_pretrained(model_name)
    return model_en, tokenizer_en


def main() -> None:
    """
    Entrypoint for the NMT demo
    """

    model_ru, tokenizer_ru = load_model("Helsinki-NLP/opus-mt-ru-en")

    input_text_ru = "Привет, как дела?"
    translated_text_en = translate(model_ru, tokenizer_ru, input_text_ru)
    print(f"Русский -> Английский: {input_text_ru} -> {translated_text_en}")

    model_en, tokenizer_en = load_model("Helsinki-NLP/opus-mt-en-ru")

    input_text_en = "Hello, how are you?"
    translated_text_ru = translate(model_en, tokenizer_en, input_text_en)
    print(f"Английский -> Русский: {input_text_en} -> {translated_text_ru}")


if __name__ == '__main__':
    main()
