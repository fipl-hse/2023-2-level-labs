"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
import lab_4_fill_words_by_ngrams.main as main_py


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = main_py.WordProcessor('<eow>')
    encoded_text = word_processor.encode(text)
    model = main_py.NGramLanguageModel(encoded_text, 2)
    model.build()
    top_p = main_py.TopPGenerator(model, word_processor, 0.5)
    top_p_result = top_p.run(51, 'Vernon')
    print(top_p_result)
    generator_types = main_py.GeneratorTypes()
    generators = {generator_types.top_p: main_py.TopPGenerator(model, word_processor, 0.5),
                  generator_types.beam_search:
                      main_py.BeamSearchTextGenerator(model, word_processor, 5)}
    quality_check = main_py.QualityChecker(generators, model, word_processor)
    quality_result = quality_check.run(100, 'The')
    print(quality_result)
    examiner = main_py.Examiner('./assets/question_and_answers.json')
    questions = examiner.provide_questions()
    students = [main_py.GeneratorRuleStudent(i, model, word_processor) for i in range(3)]
    for student in students:
        answers = student.take_exam(questions)
        result = examiner.assess_exam(answers)
        gen_type = student.get_generator_type()
        print('Type of generator:', gen_type)
        print('Answers:', ''.join(answers.values()))
        print('Accuracy:', result)
    assert result


if __name__ == "__main__":
    main()
