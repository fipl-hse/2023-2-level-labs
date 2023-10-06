"""
Checks the second lab's count tokens pairs function
"""

import unittest

import pytest

from lab_2_tokenize_by_bpe.main import collect_frequencies, count_tokens_pairs


class CountTokensPairsTest(unittest.TestCase):
    """
    Tests counting tokens pairs function
    """

    def setUp(self) -> None:
        self.word_frequencies = collect_frequencies(
            "Вез корабль карамель, наскочил корабль на мель, "
            "матросы две недели карамель на мели ели.", None, end_of_word='</s>')

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_count_tokens_pairs_ideal(self):
        """
        Ideal count tokens pairs scenario
        """
        expected = {('В', 'е'): 1, ('е', 'з'): 1, ('з', '</s>'): 1,
                    ('к', 'о'): 3, ('о', 'р'): 2, ('р', 'а'): 4,
                    ('а', 'б'): 2, ('б', 'л'): 2, ('л', 'ь'): 5,
                    ('ь', '</s>'): 3, ('к', 'а'): 2, ('а', 'р'): 2,
                    ('а', 'м'): 2, ('м', 'е'): 4, ('е', 'л'): 6,
                    ('ь', ','): 2, (',', '</s>'): 2, ('н', 'а'): 3,
                    ('а', 'с'): 1, ('с', 'к'): 1, ('о', 'ч'): 1,
                    ('ч', 'и'): 1, ('и', 'л'): 1, ('л', '</s>'): 1,
                    ('а', '</s>'): 2, ('м', 'а'): 1, ('а', 'т'): 1,
                    ('т', 'р'): 1, ('р', 'о'): 1, ('о', 'с'): 1,
                    ('с', 'ы'): 1, ('ы', '</s>'): 1, ('д', 'в'): 1,
                    ('в', 'е'): 1, ('е', '</s>'): 1, ('н', 'е'): 1,
                    ('е', 'д'): 1, ('д', 'е'): 1, ('л', 'и'): 3,
                    ('и', '</s>'): 2, ('и', '.'): 1, ('.', '</s>'): 1}

        actual = count_tokens_pairs(self.word_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_count_tokens_pairs_bad_input(self):
        """
        Count tokens pairs invalid inputs check
        """
        bad_inputs = ['string', (), None, 1, 1.1, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = count_tokens_pairs(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_count_tokens_pairs_return_value(self):
        """
        Count tokens pairs return value check
        """
        actual = count_tokens_pairs(self.word_frequencies)
        self.assertTrue(isinstance(actual, dict))
