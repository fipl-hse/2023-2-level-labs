"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             NGramLanguageModel, QualityChecker, TopPGenerator,
                                             WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as file:
        text = file.read()
    processor = WordProcessor('.')
    encoded = processor.encode(text)
    if not isinstance(encoded, tuple) or not encoded:
        raise ValueError

    model = NGramLanguageModel(encoded[:10000], 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)
    generated_text = generator.run(51, 'Vernon')
    result = generated_text
    print(result)
    print()

    types_g = GeneratorTypes()
    generators = {types_g.greedy: GreedyTextGenerator(model, processor),
            types_g.top_p: generator,
            types_g.beam_search: BeamSearchTextGenerator(model, processor, 5)}

    quality = QualityChecker(generators, model, processor)
    checks = quality.run(100, 'The')
    for check in checks:
        print(str(check))

    examiner = Examiner('./assets/question_and_answers.json')
    questions = examiner.provide_questions()
    students = []
    for i in range(3):
        students.append(GeneratorRuleStudent(i, model, processor))

    for student in students:
        stud_answers = student.take_exam(questions)
        res = examiner.assess_exam(stud_answers)

        print(student.get_generator_type(), res)
        print(' '.join(stud_answers.values()))

    assert result


if __name__ == "__main__":
    main()
