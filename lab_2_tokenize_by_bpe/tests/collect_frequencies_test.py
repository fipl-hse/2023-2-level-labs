"""
Checks the second lab's collect frequencies function
"""

import unittest
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import collect_frequencies


class CollectFrequenciesTest(unittest.TestCase):
    """
    Tests collecting frequencies function
    """
    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_ideal(self):
        """
        Ideal collect frequencies scenario
        """
        expected = {('Н', 'а', '</s>'): 1,
                    ('о', 'т', 'д', 'е', 'л', 'ь', 'н', 'ы', 'х', '</s>'): 1,
                    ('ч', 'а', 'с', 'т', 'я', 'х', '</s>'): 1,
                    ('о', 'с', 'т', 'р', 'о', 'в', 'а', '</s>'): 1,
                    ('п', 'т', 'е', 'н', 'ц', 'о', 'в', '</s>'): 2,
                    ('о', 'с', 'о', 'б', 'е', 'н', 'н', 'о', '</s>'): 1,
                    ('м', 'н', 'о', 'г', 'о', '.', '</s>'): 1,
                    ('П', 'я', 'т', 'и', 'м', 'е', 'с', 'я', 'ч', 'н', 'ы', 'е', '</s>'): 1,
                    ('а', 'л', 'ь', 'б', 'а', 'т', 'р', 'о', 'с', 'ы', '</s>'): 1,
                    ('р', 'а', 'з', 'м', 'е', 'р', 'о', 'м', '</s>'): 1,
                    ('п', 'р', 'и', 'м', 'е', 'р', 'н', 'о', '</s>'): 1,
                    ('с', '</s>'): 1, ('г', 'у', 'с', 'я', '</s>'): 1, ('т', 'а', 'к', '</s>'): 1,
                    ('п', 'л', 'о', 'т', 'н', 'о', '</s>'): 1,
                    ('з', 'а', 'с', 'е', 'л', 'я', 'ю', 'т', '</s>'): 1,
                    ('п', 'е', 'с', 'ч', 'а', 'н', 'ы', 'й', '</s>'): 1,
                    ('л', 'а', 'н', 'д', 'ш', 'а', 'ф', 'т', ',', '</s>'): 1,
                    ('ч', 'т', 'о', '</s>'): 1,
                    ('о', 'с', 'т', 'р', 'о', 'в', '</s>'): 1,
                    ('п', 'о', 'х', 'о', 'ж', '</s>'): 1,
                    ('н', 'а', '</s>'): 1,
                    ('п', 'т', 'и', 'ц', 'е', 'ф', 'е', 'р', 'м', 'у', '.', '</s>'): 1,
                    ('В', 'и', 'д', '</s>'): 1, ('у', '</s>'): 1,
                    ('с', 'е', 'й', 'ч', 'а', 'с', '</s>'): 1,
                    ('н', 'е', 'л', 'е', 'п', 'ы', 'й', ':', '</s>'): 1,
                    ('п', 'у', 'х', '</s>'): 1,
                    ('к', 'о', 'е', '-', 'г', 'д', 'е', '</s>'): 1,
                    ('н', 'а', 'ч', 'а', 'л', '</s>'): 1,
                    ('с', 'м', 'е', 'н', 'я', 'т', 'ь', 'с', 'я', '</s>'): 1,
                    ('п', 'е', 'р', 'ь', 'я', 'м', 'и', '.', '</s>'): 1}
        actual = collect_frequencies('На отдельных частях острова птенцов особенно много. '
                                     'Пятимесячные альбатросы размером примерно с гуся так плотно '
                                     'заселяют песчаный ландшафт, что остров похож на птицеферму. '
                                     'Вид у птенцов сейчас нелепый: пух кое-где начал сменяться '
                                     'перьями.', None, '</s>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_none_prepared_word(self):
        """
        Collect frequencies with None as prepare_word's return value
        """
        expected = None
        with mock.patch("lab_2_tokenize_by_bpe.main.prepare_word", return_value=None):
            actual = collect_frequencies('На отдельных частях острова '
                                         'птенцов особенно много. Пятимесячные '
                                         'альбатросы размером примерно '
                                         'с гуся так плотно заселяют песчаный '
                                         'ландшафт, что остров похож на птицеферму. '
                                         'Вид у птенцов сейчас нелепый: '
                                         'пух кое-где начал сменяться перьями.',
                                         None, '</s>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_bad_input(self):
        """
        Collect frequencies invalid inputs check
        """
        bad_inputs = [{}, (), None, 1, 1.1, True, [None]]
        start_of_word_bad_input = [{}, (), 1, 1.1, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = collect_frequencies(bad_input, None, '</s>')
            self.assertEqual(expected, actual)

            actual = collect_frequencies('На отдельных частях острова '
                                         'птенцов особенно много. Пятимесячные '
                                         'альбатросы размером примерно '
                                         'с гуся так плотно заселяют песчаный '
                                         'ландшафт, что остров похож на птицеферму. '
                                         'Вид у птенцов сейчас нелепый: '
                                         'пух кое-где начал сменяться перьями.', None, bad_input)
            self.assertEqual(expected, actual)

        for start_bad_input in start_of_word_bad_input:
            actual = collect_frequencies('На отдельных частях острова птенцов особенно много.',
                                         start_bad_input, '</s>')
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_return_value(self):
        """
        Collect frequencies return value check
        """
        text = '— Боже мой! — выдыхает Нэнси с подобающим моменту трепетом.'
        expected = 9
        actual = collect_frequencies(text, None, '</s>')
        self.assertEqual(expected, len(actual))
        for key in actual:
            self.assertTrue(isinstance(actual[key], int))
            self.assertTrue(isinstance(key, tuple))
        self.assertTrue(isinstance(actual, dict))
