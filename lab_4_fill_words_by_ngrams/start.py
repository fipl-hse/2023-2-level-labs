"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator,
                                           GreedyTextGenerator,
                                           NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent,
                                             GeneratorTypes, QualityChecker,
                                             TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor('<eow>')
    encoded = word_processor.encode(text)
    if not (isinstance(encoded, tuple) and encoded):
        raise ValueError
    lang_model = NGramLanguageModel(encoded[:10000], 2)
    lang_model.build()
    top_p_generator = TopPGenerator(lang_model, word_processor, 0.5)
    result_top = top_p_generator.run(51, 'Vernon')
    print(result_top)
    result = result_top
    generators = GeneratorTypes()
    generators_dict = {generators.greedy: GreedyTextGenerator(lang_model, word_processor),
                       generators.top_p: top_p_generator,
                       generators.beam_search: BeamSearchTextGenerator(lang_model, word_processor, 5)}
    quality_checker = QualityChecker(generators_dict, lang_model, word_processor)
    checking = quality_checker.run(100, 'The')
    for check in checking:
        print(str(check))
    examiner = Examiner('./assets/question_and_answers.json')
    questions = examiner.provide_questions()
    students = []
    for stud_id in range(3):
        students.append(GeneratorRuleStudent(stud_id, lang_model, word_processor))
    for student in students:
        answers = student.take_exam(questions)
        exam = examiner.assess_exam(answers)
        generator_type = student.get_generator_type()
        print('Type of generator is ', generator_type)
        print('Answers: ', ''.join(answers.values()))
        print('Share of the correct answers is ', str(exam))
    assert result


if __name__ == "__main__":
    main()
