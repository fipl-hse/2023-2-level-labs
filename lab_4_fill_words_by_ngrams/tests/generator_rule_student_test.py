# pylint: disable=protected-access, duplicate-code
"""
GeneratorRuleStudent class tests
"""

import unittest

import pytest

from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (GeneratorRuleStudent, GeneratorTypes, TopPGenerator,
                                             WordProcessor)


class GeneratorRuleStudentTest(unittest.TestCase):
    """
    Tests GeneratorRuleStudent class functionality
    """

    def setUp(self) -> None:
        text = '''And so the Crane came to the Fox for the dinner party.
                The Fox had cooked semolina for the dinner and smeared it over the 
                plate. Then she served it and treated her guest.'''
        self.word_processor = WordProcessor('<eos>')
        self.encoded = self.word_processor.encode(text)
        self.language_model = NGramLanguageModel(self.encoded, 2)
        self.language_model.build()
        generators_types = GeneratorTypes()
        self.generators_types = [generators_types.greedy, generators_types.beam_search,
                                 generators_types.top_p]
        self.generators = [GreedyTextGenerator(self.language_model, self.word_processor),
                           TopPGenerator(self.language_model, self.word_processor, 0.5),
                           BeamSearchTextGenerator(self.language_model, self.word_processor, 5)]

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_fields(self):
        """
        Checks if GeneratorRuleStudent fields are created correctly
        """
        for generator_type in self.generators_types:
            generator_rule_student = GeneratorRuleStudent(generator_type,
                                                          self.language_model,
                                                          self.word_processor)
            self.assertEqual(generator_type, generator_rule_student._generator_type)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_take_exam(self):
        """
        Checks GeneratorRuleStudent take_exam method ideal scenario
        """
        expected = [
            {'The fox cooked semolina for the dinner and smeared it over the plate.':
                 'Th e fox cooked semolina for the dinner and smeared it over the plate.'},
            {'The fox cooked semolina for the dinner and smeared it over the plate.':
                 'Th e fox cooked semolina for the dinner and smeared it over the plate.'},
            {'The fox cooked semolina for the dinner and smeared it over the plate.':
                 'Th e fox cooked semolina for the dinner and smeared it over the plate.'}
        ]
        for index, generator_type in enumerate(self.generators_types):
            generator_rule_student = GeneratorRuleStudent(generator_type,
                                                          self.language_model,
                                                          self.word_processor)
            actual = generator_rule_student.take_exam(
                [('The fox cooked semolina for the dinner and smeared it over the plate.', 2)]
            )
            self.assertEqual(expected[index], actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_take_exam_invalid_input(self):
        """
        Checks GeneratorRuleStudent take_exam method with invalid inputs
        """
        bad_inputs = [{}, 1, None, (), 1.1, 'string']

        for generator_type in self.generators_types:
            generator_rule_student = GeneratorRuleStudent(generator_type, self.language_model,
                                                          self.word_processor)
            with self.assertRaises(ValueError):
                for bad_input in bad_inputs:
                    generator_rule_student.take_exam(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_take_exam_empty_input(self):
        """
        Checks GeneratorRuleStudent take_exam method with invalid inputs
        """
        bad_input = []

        for generator_type in self.generators_types:
            generator_rule_student = GeneratorRuleStudent(generator_type, self.language_model,
                                                          self.word_processor)
            with self.assertRaises(ValueError):
                generator_rule_student.take_exam(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_take_exam_none_run(self):
        """
        Checks GeneratorRuleStudent take_exam method with None
        as run return value
        """
        for generator_type in self.generators_types:
            generator_rule_student = GeneratorRuleStudent(generator_type, self.language_model,
                                                          self.word_processor)

            with self.assertRaises(ValueError):
                generator_rule_student.take_exam([
                    ('The fox cooked semolina for the dinner and smeared it over the plate.', 0)])

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_generator_rule_student_get_generator_type(self):
        """
        Checks GeneratorRuleStudent get_generator_type method ideal scenario
        """
        expected = ['Greedy Generator', 'Beam Search Generator', 'Top-P Generator']
        for index, generator_type in enumerate(self.generators_types):
            generator_rule_student = GeneratorRuleStudent(generator_type,
                                                          self.language_model,
                                                          self.word_processor)
            actual = generator_rule_student.get_generator_type()
            self.assertEqual(expected[index], actual)
