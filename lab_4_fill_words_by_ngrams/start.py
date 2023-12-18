"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_4_fill_words_by_ngrams.main import (BeamSearchTextGenerator, Examiner,
                                             GeneratorRuleStudent, GeneratorTypes,
                                             NGramLanguageModel, QualityChecker, TopPGenerator,
                                             WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor('<eos>')
    encoded_text = word_processor.encode(text)
    lang_model = NGramLanguageModel(encoded_text, 2)
    lang_model.build()
    top_p_generator = TopPGenerator(lang_model, word_processor, 0.5)

    generator_types = GeneratorTypes()
    generators = {generator_types.top_p: TopPGenerator(lang_model, word_processor, 0.5),
                  generator_types.beam_search: BeamSearchTextGenerator(lang_model,
                                                                       word_processor, 5)}
    checker = QualityChecker(generators, lang_model, word_processor)

    examiner = Examiner("/Users/alesamaskovceva/Documents/python/2023-2-level-labs/lab_4_fill_words_by_ngrams"
                        "/assets/question_and_answers.json")
    questions = examiner.provide_questions()
    students = [GeneratorRuleStudent(generator_types.greedy, lang_model, word_processor),
                GeneratorRuleStudent(generator_types.top_p, lang_model, word_processor),
                GeneratorRuleStudent(generator_types.beam_search, lang_model, word_processor)]
    answers = {student: student.take_exam(questions) for student in students}
    assessment = {student: examiner.assess_exam(answer) for student, answer in answers.items()}
    result = ""
    for student, accuracy in assessment.items():
        result += f"{student.get_generator_type()}: {accuracy}\n"
    print(result)

    assert result


if __name__ == "__main__":
    main()
