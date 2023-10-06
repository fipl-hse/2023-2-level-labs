"""
Checks the second lab's merge tokens function
"""

import unittest

import pytest

from lab_2_tokenize_by_bpe.main import merge_tokens


class MergeTokensTest(unittest.TestCase):
    """
    Tests merging tokens function
    """
    def setUp(self) -> None:
        self.word_frequencies = {
            ('В', 'е', 'з', '</s>'): 1, ('к', 'о', 'р', 'а', 'б', 'л', 'ь', '</s>'): 2,
            ('к', 'а', 'р', 'а', 'м', 'е', 'л', 'ь', ',', '</s>'): 1,
            ('н', 'а', 'с', 'к', 'о', 'ч', 'и', 'л', '</s>'): 1,
            ('н', 'а', '</s>'): 2, ('м', 'е', 'л', 'ь', ',', '</s>'): 1,
            ('м', 'а', 'т', 'р', 'о', 'с', 'ы', '</s>'): 1,
            ('д', 'в', 'е', '</s>'): 1, ('н', 'е', 'д', 'е', 'л', 'и', '</s>'): 1,
            ('к', 'а', 'р', 'а', 'м', 'е', 'л', 'ь', '</s>'): 1,
            ('м', 'е', 'л', 'и', '</s>'): 1, ('е', 'л', 'и', '.', '</s>'): 1}
        self.pair = ('е', 'л')

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_merge_tokens_ideal(self):
        """
        Ideal merge tokens scenario
        """
        expected = {('В', 'е', 'з', '</s>'): 1, ('к', 'о', 'р', 'а', 'б', 'л', 'ь', '</s>'): 2,
                    ('к', 'а', 'р', 'а', 'м', 'ел', 'ь', ',', '</s>'): 1,
                    ('н', 'а', 'с', 'к', 'о', 'ч', 'и', 'л', '</s>'): 1,
                    ('н', 'а', '</s>'): 2, ('м', 'ел', 'ь', ',', '</s>'): 1,
                    ('м', 'а', 'т', 'р', 'о', 'с', 'ы', '</s>'): 1,
                    ('д', 'в', 'е', '</s>'): 1, ('н', 'е', 'д', 'ел', 'и', '</s>'): 1,
                    ('к', 'а', 'р', 'а', 'м', 'ел', 'ь', '</s>'): 1, ('м', 'ел', 'и', '</s>'): 1,
                    ('ел', 'и', '.', '</s>'): 1}
        actual = merge_tokens(self.word_frequencies, self.pair)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_merge_tokens_bad_input(self):
        """
        Merge tokens invalid inputs check
        """
        word_frequencies_bad_input = ['string', (), None, 1, 1.1, True, [None]]
        pair_bad_input = ['string', None, 1, 1.1, True, [None], {}]
        expected = None
        for index, bad_input in enumerate(word_frequencies_bad_input):
            actual = merge_tokens(bad_input, self.pair)
            self.assertEqual(expected, actual)

            actual = merge_tokens(self.word_frequencies, pair_bad_input[index])
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_merge_tokens_return_value(self):
        """
        Merge tokens return value check
        """
        actual = merge_tokens(self.word_frequencies, self.pair)
        for key in actual:
            self.assertTrue(isinstance(actual[key], int))
            self.assertTrue(isinstance(key, tuple))
        self.assertTrue(isinstance(actual, dict))
