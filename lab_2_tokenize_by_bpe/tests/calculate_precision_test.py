"""
Checks the second lab's calculate precision function
"""
import unittest

import pytest

from lab_2_tokenize_by_bpe.main import calculate_precision


class CalculatePrecisionTest(unittest.TestCase):
    """
    Tests calculating precision function
    """
    def setUp(self) -> None:
        self.actual_nrgams = [('Д',), ('о',), ('б',), ('р',), ('ы',), ('й',),
                               (' ',), ('в',), ('е',), ('ч',), ('е',), ('р',),
                               ('!',), (' ',), ('К',), ('а',), ('к',), (' ',),
                               ('п',), ('р',), ('о',), ('ш',), ('е',), ('л',),
                               (' ',), ('В',), ('а',), ('ш',), (' ',), ('д',),
                               ('е',), ('н',), ('ь',), ('?',)]
        self.reference_ngrams = [('З',), ('д',), ('р',), ('а',), ('в',), ('с',),
                                  ('т',), ('в',), ('у',), ('й',), ('т',), ('е',),
                                  ('!',), (' ',), ('К',), ('а',), ('к',), (' ',),
                                  ('п',), ('р',), ('о',), ('ш',), ('е',), ('л',),
                                  (' ',), ('В',), ('а',), ('ш',), (' ',), ('д',),
                                  ('е',), ('н',), ('ь',), ('?',)]

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_calculate_precision_ideal(self):
        """
        Ideal calculate precision scenario
        """
        expected = 0.8181818181818182
        actual = calculate_precision(self.actual_nrgams, self.reference_ngrams)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_calculate_precision_missing_value(self):
        """
        Calculate precision missing value check
        """
        expected = 0.0
        actual = calculate_precision([], self.reference_ngrams)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_calculate_precision_bad_input(self):
        """
        Calculate precision invalid inputs check
        """
        actual_ngrams_bad_input = [(), 'string', {}, None, 1, 1.1, True]
        reference_ngrams_bad_input = [None, (), 1.1, True, 1, 'string', {}]
        expected = None
        for index, bad_input in enumerate(actual_ngrams_bad_input):
            actual = calculate_precision(bad_input, self.reference_ngrams)
            self.assertEqual(expected, actual)

            actual = calculate_precision(self.actual_nrgams, reference_ngrams_bad_input[index])
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_calculate_precision_return_value(self):
        """
        Calculate precision return value check
        """
        actual = calculate_precision(self.actual_nrgams, self.reference_ngrams)
        self.assertTrue(isinstance(actual, float))
