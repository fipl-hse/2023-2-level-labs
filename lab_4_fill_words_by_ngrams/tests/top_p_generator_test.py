# pylint: disable=protected-access, duplicate-code
"""
TopPGenerator class tests
"""

import unittest

import pytest

from lab_3_generate_by_ngrams.main import NGramLanguageModel
from lab_4_fill_words_by_ngrams.main import TopPGenerator, WordProcessor


class TopPGeneratorTest(unittest.TestCase):
    """
    Tests TopPGenerator class functionality
    """

    def setUp(self) -> None:
        text = '''Then she served it and treated her guest. The Crane went
                peck-peck with his beak, knocked and knocked but couldn’t pick even
                a bit of fare. The Fox kept licking the cereal until she had eaten
                it all. When there’s no cereal at all, the Fox said,
                «Don’t feel offended, buddy. There’s nothing more to treat you”'''
        self.word_processor = WordProcessor('<eos>')
        self.encoded = self.word_processor.encode(text)
        self.language_model = NGramLanguageModel(self.encoded, 2)
        self.language_model.build()

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_fields(self):
        """
        Checks if TopPGenerator fields are created correctly
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        self.assertEqual(self.language_model, generator._model)
        self.assertEqual(self.word_processor, generator._word_processor)
        self.assertEqual(0.5, generator._p_value)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_sorting_prob(self):
        """
        Checks TopPGenerator run method ideal scenario with correct sorting
        of the probabilities
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.2)
        expected = 'The fox.'
        actual = generator.run(1, 'the')
        self.assertIn(actual, expected)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_sorting_candidates(self):
        """
        Checks TopPGenerator run method ideal scenario with correct sorting
        of the next candidates
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.2)
        expected = 'Fox said.'
        actual = generator.run(1, 'Fox')
        self.assertIn(actual, expected)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_invalid_input(self):
        """
        Checks TopPGenerator run method with invalid inputs
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        seq_len_bad_input = [[None], {}, None, (), 1.1, 'string']
        prompt_bad_input = [1, [None], {}, None, (), 1.1]

        with self.assertRaises(ValueError):
            for bad_input in seq_len_bad_input:
                generator.run(bad_input, 'dinner')

            for bad_input in prompt_bad_input:
                generator.run(10, bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_empty_input(self):
        """
        Checks TopPGenerator run method with invalid inputs
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        prompt_bad_input = ''

        with self.assertRaises(ValueError):
            generator.run(int('string'), 'dinner')

        with self.assertRaises(ValueError):
            generator.run(10, prompt_bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_invalid_seq_len(self):
        """
        Checks TopPGenerator run method with invalid seq_len
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)

        with self.assertRaises(ValueError):
            generator.run(-1, 'dinner')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_none_encode(self):
        """
        Checks TopPGenerator run method with None as encode return value
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        generator._word_processor.encode = lambda x: None

        with self.assertRaises(ValueError):
            generator.run(10, 'dinner')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_none_decode(self):
        """
        Checks TopPGenerator run method with None as decode return value
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        generator._word_processor.decode = lambda x: None

        with self.assertRaises(ValueError):
            generator.run(10, 'dinner')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_none_generate_next_token(self):
        """
        Checks GreedyTextGenerator run method with None
        as generate_next_token return value
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        generator._model.generate_next_token = lambda sequence: None
        with self.assertRaises(ValueError):
            generator.run(10, 'dinner')

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run_empty_next_candidate(self):
        """
        Checks GreedyTextGenerator run method with empty next candidate
        """
        generator = TopPGenerator(self.language_model, self.word_processor, 0.5)
        generator._model.generate_next_token = lambda sequence: {}
        actual = generator.run(10, 'dinner')
        expected = 'Dinner.'
        self.assertEqual(expected, actual)
