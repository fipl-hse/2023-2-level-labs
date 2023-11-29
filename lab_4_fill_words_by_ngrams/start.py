"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from config.constants import PROJECT_ROOT
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             NGramLanguageModel, QualityChecker, TopPGenerator,
                                             WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = WordProcessor('<eow>')
    encoded = processor.encode(text)
    model = NGramLanguageModel(encoded, 2)
    generator = TopPGenerator(model, processor, 0.5)
    result = generator.run(51, 'Vernon')
    print(result)

    gen_types = GeneratorTypes()
    gens = {gen_types.greedy: GreedyTextGenerator(model, processor),
            gen_types.top_p: generator,
            gen_types.beam_search: BeamSearchTextGenerator(model, processor, 5)}
    quality_checker = QualityChecker(gens, model, processor)
    print(quality_checker)  # run has None => raises ValueError
    # checks = quality_checker.run(100, 'The')
    # for check in checks:
    #     result = str(check)
    #     print(result)

    students = []
    for student_id in range(3):
        students.append(GeneratorRuleStudent(student_id, model, processor))
    json_path = str(PROJECT_ROOT / 'lab_4_fill_words_by_ngrams' / 'assets' /
                    'question_and_answers.json')
    examiner = Examiner(json_path)

    for student in students:
        # ValueError...
        student_answers = student.take_exam(examiner.provide_questions())
        score = examiner.assess_exam(student_answers)
        print(student, score)

    assert result


if __name__ == "__main__":
    main()
