# pylint: disable=protected-access
"""
WordProcessor class tests
"""

import unittest

import pytest

from lab_3_generate_by_ngrams.main import TextProcessor
from lab_4_fill_words_by_ngrams.main import WordProcessor


class WordProcessorTest(unittest.TestCase):
    """
    Tests WordProcessor class functionality
    """

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_inherits_from_text_processor(self):
        """
        Tests WordProcessor class functionality
        """
        self.assertTrue(issubclass(WordProcessor, TextProcessor))

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_fields(self):
        """
        Checks if WordProcessor fields are created correctly
        """
        word_processor = WordProcessor('<eos>')
        self.assertEqual('<eos>', word_processor._end_of_word_token)
        self.assertEqual({'<eos>': 0}, word_processor._storage)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize(self):
        """
        Checks WordProcessor _tokenize method ideal scenario
        """
        word_processor = WordProcessor('<eos>')
        expected = ('she', 'is', 'happy', '<eos>', 'he', 'is', 'happy', '<eos>')
        actual = word_processor._tokenize('She is happy. He is happy.')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize_invalid_input(self):
        """
        Checks WordProcessor _tokenize method bad scenario
        """
        word_processor = WordProcessor('<eos>')
        bad_inputs = [(), [None], {}, None, 1, 1.1, True]

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                word_processor._tokenize(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize_empty_input(self):
        """
        Checks WordProcessor _tokenize method bad scenario
        """
        word_processor = WordProcessor('<eos>')
        bad_input = ''

        with self.assertRaises(ValueError):
            word_processor._tokenize(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_put(self):
        """
        Checks WordProcessor _put method ideal scenario
        """
        word_processor = WordProcessor('<eos>')
        word_processor._put('she')
        actual = word_processor._storage
        self.assertEqual({'<eos>': 0, 'she': 1}, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_put_invalid_input(self):
        """
        Checks WordProcessor _put method with invalid inputs
        """
        word_processor = WordProcessor('<eos>')
        bad_inputs = [(), [None], {}, None, 1, 1.1, True]

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                word_processor._put(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_put_empty_input(self):
        """
        Checks WordProcessor _put method with invalid inputs
        """
        word_processor = WordProcessor('<eos>')
        bad_input = ''

        with self.assertRaises(ValueError):
            word_processor._put(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_text(self):
        """
        Checks WordProcessor _postprocess_decoded_text method ideal scenario
        """
        word_processor = WordProcessor('<eos>')
        encoded = word_processor.encode('She is HAPPY! He is HAPPY!')
        protected_decoded = word_processor._decode(encoded)
        expected = 'She is happy. He is happy.'
        actual = word_processor._postprocess_decoded_text(protected_decoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_sentence(self):
        """
        Checks WordProcessor _postprocess_decoded_text method ideal scenario
        """
        word_processor = WordProcessor('<eos>')
        encoded = word_processor.encode('She is HAPPY')
        protected_decoded = word_processor._decode(encoded)
        expected = 'She is happy.'
        actual = word_processor._postprocess_decoded_text(protected_decoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_text_invalid_input(self):
        """
        Checks WordProcessor _postprocess_decoded_text method with invalid inputs
        """
        word_processor = WordProcessor('<eos>')
        bad_inputs = [1, [None], {}, None, 1.1, True, 'string']

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                word_processor._postprocess_decoded_text(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_text_empty_input(self):
        """
        Checks WordProcessor _postprocess_decoded_text method with invalid inputs
        """
        word_processor = WordProcessor('<eos>')
        bad_input = ()

        with self.assertRaises(ValueError):
            word_processor._postprocess_decoded_text(bad_input)
