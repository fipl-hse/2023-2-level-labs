"""
Checks the second lab's encode function
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import encode


class EncodeTest(unittest.TestCase):
    """
    Tests encoding function
    """

    def setUp(self) -> None:
        path_to_tests_directory = Path(__file__).parent
        with open(path_to_tests_directory / 'vocabulary.json', 'r', encoding='utf-8') as json_file:
            self.vocabulary = json.load(json_file)

        self.ideal_original_text = ('Активный взмах крыльями альбатрос делает только при взлете, '
                                    'полагаясь далее на силу и направление ветра.')
        self.original_text_with_unk = ('Под влиянием латинского albus («белый») alcatraz'
                                       ' чуть позднее превратился в albatross.')
        self.arbitrary_text = '你好！我是俄罗斯人。我住在下诺夫哥罗德。我喜欢程序设计。'

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_ideal(self):
        """
        Ideal encode scenario
        """
        expected = [144, 179, 95, 171, 79, 178, 34, 171, 176, 181,
                    169, 190, 34, 179, 89, 72, 200, 73, 34, 0, 186,
                    34, 55, 68, 60, 34, 96, 72, 67, 34, 40, 34, 171,
                    176, 180, 60, 174, 104, 34, 83, 68, 172, 169, 200,
                    186, 197, 34, 54, 69, 174, 34, 75, 34, 91, 180, 188,
                    34, 177, 34, 75, 184, 84, 171, 180, 37, 174, 34, 171,
                    60, 84, 106, 34]
        actual = encode(self.ideal_original_text, self.vocabulary, None,
                        '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_with_unk(self):
        """
        Encode scenario with unknown tokens
        """
        expected = [158, 81, 34, 171, 70, 200, 77, 174, 181, 34, 180, 47, 63,
                    43, 53, 34, 129, 134, 16, 16, 139, 34, 101, 142, 170, 174,
                    180, 196, 178, 143, 102, 34, 129, 134, 130, 129, 140, 138,
                    129, 16, 34, 192, 188, 187, 197, 34, 83, 176, 173, 76, 174,
                    34, 184, 85, 171, 185, 47, 177, 180, 186, 200, 34, 171, 34,
                    129, 134, 16, 129, 140, 138, 137, 139, 139, 106, 34]
        actual = encode(self.original_text_with_unk, self.vocabulary, None,
                        '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_arbitrary_text(self):
        """
        Encode scenario for absolutely arbitrary text
        """
        expected = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 34]
        actual = encode(self.arbitrary_text, self.vocabulary, None,
                        '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_none_prepare_word(self):
        """
        Encode with None as prepare_word's return value
        """
        expected = None
        with mock.patch("lab_2_tokenize_by_bpe.main.prepare_word", return_value=None):
            actual = encode(self.arbitrary_text, self.vocabulary, None,
                            '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_none_tokenize_word(self):
        """
        Encode with None as tokenize_word's return value
        """
        expected = None
        with mock.patch("lab_2_tokenize_by_bpe.main.tokenize_word", return_value=None):
            actual = encode(self.arbitrary_text, self.vocabulary, None,
                            '</s>', '<unk>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_bad_input(self):
        """
        Encode invalid inputs check
        """
        original_text_bad_input = [(), [None], {}, None, 1, 1.1, True]
        vocabulary_bad_input = [None, (), 1.1, True, [None], 'string', 1]
        start_end_bad_input = [(), {}, 1, 1.1, True, [None]]
        unknown_bad_input = [(), {}, None, 1, 1.1, True, [None]]
        expected = None
        for index, bad_input in enumerate(original_text_bad_input):
            actual = encode(bad_input, self.vocabulary, None,
                            '</s>', '<unk>')
            self.assertEqual(expected, actual)

            actual = encode('Активный взмах крыльями альбатрос делает только при взлете, '
                            'полагаясь далее на силу и направление ветра.',
                            vocabulary_bad_input[index], None,
                            '</s>', '<unk>')
            self.assertEqual(expected, actual)

            actual = encode('Активный взмах крыльями альбатрос делает только при взлете, '
                            'полагаясь далее на силу и направление ветра.',
                            self.vocabulary, None,
                            '</s>', unknown_bad_input[index])
            self.assertEqual(expected, actual)

        for bad_input in start_end_bad_input:
            actual = encode('Активный взмах крыльями альбатрос делает только при взлете, '
                            'полагаясь далее на силу и направление ветра.',
                            self.vocabulary, bad_input, '</s>', '<unk>')
            self.assertEqual(expected, actual)

            actual = encode('Активный взмах крыльями альбатрос делает только при взлете, '
                            'полагаясь далее на силу и направление ветра.',
                            self.vocabulary, None, bad_input, '<unk>')
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_encode_return_value(self):
        """
        Encode return value check
        """
        actual = encode(self.ideal_original_text, self.vocabulary, None,
                        '</s>', '<unk>')
        self.assertTrue(isinstance(actual, list))
