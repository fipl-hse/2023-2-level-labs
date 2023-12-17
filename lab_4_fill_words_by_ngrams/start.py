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
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = WordProcessor('.')
    encoded = processor.encode(text)

    if not (isinstance(encoded, tuple) and encoded):
        raise ValueError

    model = NGramLanguageModel(encoded[:10000], 2)
    model.build()
    generator = TopPGenerator(model, processor, 0.5)

    generated = generator.run(51, 'Vernon')
    print(generated)
    result = generated

    generators = GeneratorTypes()
    generators_dict = {generators.greedy: GreedyTextGenerator(
        model, processor), generators.top_p: generator, generators.beam_search:
        BeamSearchTextGenerator(model, processor, 5)}

    quality = QualityChecker(generators_dict, model, processor)
    perplexity = quality.run(100, 'The')
    for score in perplexity:
        print(str(score))

    examiner = Examiner('./assets/question_and_answers.json')
    questions = examiner.provide_questions()

    list_of_students = []

    for i in range(3):
        list_of_students.append(GeneratorRuleStudent(i, model, processor))

    for student in list_of_students:
        answers = student.take_exam(questions)
        exam = examiner.assess_exam(answers)
        gen = student.get_generator_type()
        result = str(exam)

        print(gen, exam)
        print(' '.join(answers.values()))

    assert result


if __name__ == "__main__":
    main()
