"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator,
                                            GreedyTextGenerator, NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (GeneratorTypes, TopPGenerator, 
                                            QualityChecker, WordProcessor)

def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None

    word_processor = WordProcessor(end_of_word_token='.')
    encoded_text = word_processor.encode(text)

    lang_model = NGramLanguageModel(encoded_text, 2)
    lang_model.build()

    top_p_gen = TopPGenerator(lang_model, word_processor, 0.5)
    result = top_p_gen.run(51, 'Vernon')
    #print(result, '\n')

    gen_types = GeneratorTypes()
    gen_dict = {gen_types.greedy: GreedyTextGenerator(lang_model, word_processor),
            gen_types.top_p: top_p_gen,
            gen_types.beam_search: BeamSearchTextGenerator(lang_model, word_processor, 5)}

    quality_checker = QualityChecker(gen_dict, lang_model, word_processor)
    run_check = quality_checker.run(100, 'The')
    for current_check in run_check:
        print(current_check)
    assert result
if __name__ == "__main__":
    main()
