# pylint: disable=protected-access
"""
QualityChecker class tests
"""

import unittest

import pytest

from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel)
from lab_4_fill_words_by_ngrams.main import (GenerationResultDTO, GeneratorTypes, QualityChecker,
                                             TopPGenerator, WordProcessor)


class QualityCheckerTest(unittest.TestCase):
    """
    Tests QualityChecker class functionality
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
        self.generators = {
            generators_types.greedy: GreedyTextGenerator(self.language_model, self.word_processor),
            generators_types.top_p: TopPGenerator(self.language_model, self.word_processor, 0.5),
            generators_types.beam_search: BeamSearchTextGenerator(self.language_model,
                                                                  self.word_processor, 5)
        }

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_fields(self):
        """
        Checks if QualityChecker fields are created correctly
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        self.assertEqual(self.generators, checker._generators)
        self.assertEqual(self.language_model, checker._language_model)
        self.assertEqual(self.word_processor, checker._word_processor)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity(self):
        """
        Checks QualityChecker calculate_perplexity method ideal scenario
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        expected = 2.449489742783178
        actual = checker._calculate_perplexity('Dinner and so the plate.')
        self.assertAlmostEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_invalid_input(self):
        """
        Checks QualityChecker calculate_perplexity method with invalid inputs
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        bad_inputs = [(), [None], {}, None, 1, 1.1, True]

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                checker._calculate_perplexity(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_empty_input(self):
        """
        Checks QualityChecker calculate_perplexity method with invalid inputs
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        bad_input = ''

        with self.assertRaises(ValueError):
            checker._calculate_perplexity(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_none_encode_for_text(self):
        """
        Checks QualityChecker calculate_perplexity method with None as encode return value
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)

        with self.assertRaises(ValueError):
            checker._calculate_perplexity(' ')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_none_encode_for_prompt(self):
        """
        Checks QualityChecker calculate_perplexity method with None as encode return value
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)

        with self.assertRaises(ValueError):
            checker._calculate_perplexity('1 the')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_none_probabilities(self):
        """
        Checks QualityChecker calculate_perplexity method with None as probabilities return value
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        self.language_model.generate_next_token = lambda x: None

        with self.assertRaises(ValueError):
            checker._calculate_perplexity('Dinner and so the plate.')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_calculate_perplexity_nothing_to_generate(self):
        """
        Checks QualityChecker calculate_perplexity method, when the number of
        tokens to generate is 0
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        with self.assertRaises(ValueError):
            checker._calculate_perplexity('.')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_run(self):
        """
        Checks QualityChecker run method ideal scenario
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        actual = checker.run(1, 'Dinner')
        for class_object in actual:
            self.assertTrue(isinstance(class_object, GenerationResultDTO))

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_run_invalid_input(self):
        """
        Checks QualityChecker run method with invalid inputs
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        seq_len_bad_input = [[None], {}, None, (), 1.1, 'string']
        prompt_bad_input = [1, [None], {}, None, (), 1.1]

        with self.assertRaises(ValueError):
            for bad_input in seq_len_bad_input:
                checker.run(bad_input, 'Dinner')

            for bad_input in prompt_bad_input:
                checker.run(10, bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_run_empty_input(self):
        """
        Checks QualityChecker run method with invalid inputs
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        prompt_bad_input = ''

        with self.assertRaises(ValueError):
            checker.run(int('string'), 'Dinner')

        with self.assertRaises(ValueError):
            checker.run(10, prompt_bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_run_invalid_seq_len(self):
        """
        Checks QualityChecker run method with invalid seq_len
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)

        with self.assertRaises(ValueError):
            checker.run(-1, 'Dinner')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_quality_checker_run_none_perplexity(self):
        """
        Checks QualityChecker run method with None as perplexity return value
        """
        checker = QualityChecker(self.generators, self.language_model,
                                 self.word_processor)
        checker._calculate_perplexity = lambda x: None
        with self.assertRaises(ValueError):
            checker.run(10, 'Dinner')
