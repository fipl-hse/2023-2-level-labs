"""
Filling word by ngrams starter
"""
# pylint:disable=too-many-locals,unused-import
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             NGramLanguageModel, QualityChecker,
                                             TopPGenerator, WordProcessor)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_proc = WordProcessor('<eow>')
    encoded = word_proc.encode(text)
    lang_model = NGramLanguageModel(encoded, n_gram_size=2)
    lang_model.build()
    top_p = TopPGenerator(language_model=lang_model, word_processor=word_proc, p_value=0.5)
    generated_text = top_p.run(51, "Vernon")
    print(generated_text)

    generators = GeneratorTypes()
    generators_dict = {generators.greedy: GreedyTextGenerator(lang_model, word_proc),
                       generators.top_p: top_p,
                       generators.beam_search: BeamSearchTextGenerator(lang_model, word_proc, 5)}
    quality_check = QualityChecker(generators_dict, lang_model, word_proc)
    run_check = quality_check.run(100, 'The')
    result_quality_check = [str(cur_check) for cur_check in run_check]
    print("\n".join(result_quality_check))
    examiner = Examiner('C:/Users/User/Desktop/hse/programming/2023-2-level-labs/lab_4_fill_words_by_ngrams/assets'
                        '/question_and_answers.json')
    tasks = examiner.provide_questions()
    students = [GeneratorRuleStudent(i, lang_model, word_proc) for i in range(3)]
    answers = {student: student.take_exam(tasks) for student in students}
    assessment = {student: examiner.assess_exam(answer) for student, answer in answers.items()}
    result = ""
    for student, accuracy in assessment.items():
        result += f"Accuracy of student ({student.get_generator_type()}): {accuracy}\n"
    print(result)
    assert result


if __name__ == "__main__":
    main()
