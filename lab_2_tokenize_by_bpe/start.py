"""
BPE Tokenizer starter
"""
import json
from pathlib import Path

from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation
    """
    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)
    if not (isinstance(encoded_text, tuple) and encoded_text):
        return
    result = text_processor.decode(encoded_text)
    print(result)

    language_model = NGramLanguageModel(encoded_text[:100], 3)
    result = language_model.build()
    print(result)

    other_language_model = NGramLanguageModel(encoded_text, 7)
    greedy_generator = GreedyTextGenerator(other_language_model, text_processor)
    result = greedy_generator.run(51, "Vernon")

    print(result)


if __name__ == "__main__":
    main()
