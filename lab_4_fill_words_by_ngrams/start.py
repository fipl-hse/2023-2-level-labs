"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             QualityChecker, TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor('<eow>')
    encoded = word_processor.encode(text)
    lang_model = NGramLanguageModel(encoded, n_gram_size=2)
    lang_model.build()
    top_p = TopPGenerator(lang_model, word_processor, 0.5)
    generated_text_6 = top_p.run(51, 'Vernon')
    print(generated_text_6)
    gen_types = GeneratorTypes()
    generators = {gen_types.greedy: GreedyTextGenerator(lang_model, word_processor),
                  gen_types.top_p: top_p,
                  gen_types.beam_search: BeamSearchTextGenerator(lang_model, word_processor, 5)}
    quality_check = QualityChecker(generators, lang_model, word_processor)
    generating = quality_check.run(100, 'The')
    result_8 = [str(current) for current in generating]
    print("\n".join(result_8))
    examiner = Examiner('./assets/question_and_answers.json')
    questions = examiner.provide_questions()
    students = []
    for student_id in range(3):
        students.append(GeneratorRuleStudent(student_id, lang_model, word_processor))
    for student in students:
        answers = student.take_exam(questions)
        result = examiner.assess_exam(answers)
        generator_type = student.get_generator_type()
        print('Type of generator is ', generator_type)
        print('Answers: ', ''.join(answers.values()))
        print('Share of the correct answers is ', str(result))
    assert result


if __name__ == "__main__":
    main()
