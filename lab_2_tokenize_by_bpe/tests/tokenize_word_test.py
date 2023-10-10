"""
Checks the second lab's tokenize word function
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_2_tokenize_by_bpe.main import tokenize_word


class TokenizeWordTest(unittest.TestCase):
    """
    Tests tokenizing word function
    """
    def setUp(self) -> None:
        path_to_tests_directory = Path(__file__).parent
        with open(path_to_tests_directory / 'vocabulary.json', 'r', encoding='utf-8') as json_file:
            self.vocabulary = json.load(json_file)

        self.ideal_word = ('а', 'л', 'ь', 'б', 'а', 'т', 'р', 'о', 'с', 'ы', '</s>')
        self.word_with_unk = ('a', 'l', 'c', 'a', 't', 'r', 'a', 'z', '</s>')

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_tokenize_word_ideal(self):
        """
        Ideal tokenize word scenario
        """
        expected = [0, 186, 29]  # 196, 34
        actual = tokenize_word(self.ideal_word, self.vocabulary, '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_tokenize_word_with_unk(self):
        """
        Tokenize word scenario with unknown token
        """
        expected = [129, 134, 130, 129, 140, 138, 129, 16, 34]
        actual = tokenize_word(self.word_with_unk, self.vocabulary, '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_tokenize_word_bad_input(self):
        """
        Tokenize word invalid inputs check
        """
        word_bad_input = ['string', [None], {}, None, 1, 1.1, True]
        vocabulary_bad_input = [None, (), 1.1, True, [None], 'string', 1]
        end_of_word_bad_input = [(), {}, 1, 1.1, True, [None]]
        unknown_bad_input = [(), {}, None, 1, 1.1, True, [None]]
        expected = None
        for index, bad_input in enumerate(word_bad_input):
            actual = tokenize_word(bad_input, self.vocabulary, '</s>', '<unk>')
            self.assertEqual(expected, actual)

            actual = tokenize_word(self.ideal_word, vocabulary_bad_input[index],
                                   '</s>', '<unk>')
            self.assertEqual(expected, actual)

            actual = tokenize_word(self.ideal_word, self.vocabulary,
                                   '</s>', unknown_bad_input[index])
            self.assertEqual(expected, actual)

        for bad_input in end_of_word_bad_input:
            actual = tokenize_word(self.ideal_word, self.vocabulary,
                                   bad_input, '<unk>')
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_tokenize_word_return_value(self):
        """
        Tokenize word return value check
        """
        actual = tokenize_word(self.ideal_word, self.vocabulary, '</s>', '<unk>')
        self.assertTrue(isinstance(actual, list))
