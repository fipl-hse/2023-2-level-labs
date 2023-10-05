"""
Checks the second lab's train function
"""

import unittest
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import collect_frequencies, train


class TrainTest(unittest.TestCase):
    """
    Tests training function
    """
    def setUp(self) -> None:
        self.word_frequencies = collect_frequencies(
            'Вез корабль карамель, наскочил корабль на мель, '
            'матросы две недели карамель на мели ели.', None, '</s>')

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_train_ideal(self):
        """
        Ideal train scenario
        """
        expected = {('В', 'е', 'з', '</s>'): 1, ('корабль</s>',): 2,
                    ('карамель,</s>',): 1, ('на', 'с', 'ко', 'ч', 'и', 'л', '</s>'): 1,
                    ('на</s>',): 2, ('мель,</s>',): 1,
                    ('м', 'а', 'т', 'р', 'о', 'с', 'ы', '</s>'): 1, ('д', 'ве</s>'): 1,
                    ('недели</s>',): 1, ('карамель</s>',): 1, ('мели</s>',): 1,
                    ('ели.</s>',): 1}
        actual = train(self.word_frequencies, 30)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_train_none_word_frequencies(self):
        """
        Train with None as word_frequencies' return value
        """
        expected = None
        with mock.patch("lab_2_tokenize_by_bpe.main.merge_tokens", return_value=None):
            actual = train(self.word_frequencies, 30)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_train_bad_input(self):
        """
        Train invalid inputs check
        """
        word_frequencies_bad_input = ['string', (), 1, 1.1, True, [None]]
        num_merges_bad_input = ['string', None, (), 1.1, [None], {}]
        expected = None
        for bad_input in word_frequencies_bad_input:
            actual = train(bad_input, 30)
            self.assertEqual(expected, actual)
        for bad_input in num_merges_bad_input:
            actual = train(self.word_frequencies, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_train_return_value(self):
        """
        Train return value check
        """
        actual = train(self.word_frequencies, 100)
        for key in actual:
            self.assertTrue(isinstance(actual[key], int))
            self.assertTrue(isinstance(key, tuple))
        self.assertTrue(isinstance(actual, dict))
