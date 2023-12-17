"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes, NGramLanguageModel, QualityChecker,
                                             TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = WordProcessor('<eow>')
    encoded = processor.encode(text)
    model = NGramLanguageModel(encoded, 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)
    result = generator.run(51, 'Vernon')
    print(result, '\n')
    gen_types = GeneratorTypes()
    gens = {gen_types.greedy: GreedyTextGenerator(model, processor),
            gen_types.top_p: generator,
            gen_types.beam_search: BeamSearchTextGenerator(model, processor, 5)}
    quality_checker = QualityChecker(gens, model, processor)
    checks = quality_checker.run(100, 'The')
    for check in checks:
        print(check)
    path = 'assets/question_and_answers.json'
    examiner = Examiner(f'{path}')
    tasks = examiner.provide_questions()
    students = [GeneratorRuleStudent(i, model, processor) for i in range(3)]
    answers = {student: student.take_exam(tasks) for student in students}
    assessment = {student: examiner.assess_exam(answer) for student, answer in answers.items()}
    result = ""
    for student, accuracy in assessment.items():
        result += f"Accuracy of student ({student.get_generator_type()}): {accuracy}\n"
    print(result)
    assert result


if __name__ == "__main__":
    main()
