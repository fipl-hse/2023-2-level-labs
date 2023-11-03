# pylint: disable=protected-access
"""
Checks the third lab's GreedyTextGenerator class
"""
import unittest
from pathlib import Path

import pytest

from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


class GreedyTextGeneratorTest(unittest.TestCase):
    """
    Tests GreedyTextGenerator class functionality
    """

    def setUp(self) -> None:
        path_to_test_directory = Path(__file__).parent.parent
        with open(path_to_test_directory / 'assets' /
                  'Anna Karenina - Chapter 1.txt', 'r', encoding='utf-8') as text_file:
            text = text_file.read()
        self.processor = TextProcessor('_')
        self.encoded = self.processor.encode(text)
        self.language_model = NGramLanguageModel(self.encoded, 3)
        self.language_model.build()

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_fields(self):
        """
        Checks if GreedyTextGenerator fields are created correctly
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        self.assertEqual(self.language_model, greedy_text_generator._model)
        self.assertEqual(self.processor, greedy_text_generator._text_processor)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run(self):
        """
        Checks GreedyTextGenerator run method ideal scenario
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        actual = greedy_text_generator.run(70, 'This')
        self.assertEqual('This the his the his the his the '
                         'his the his the his the his the his the h.', actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_none_encode(self):
        """
        Checks GreedyTextGenerator run method with None as encode return value
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._text_processor.encode = lambda x: None
        actual = greedy_text_generator.run(70, 'This')
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_none_generate_next_token(self):
        """
        Checks GreedyTextGenerator run method with None
        as generate_next_token return value
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._model.generate_next_token = lambda sequence: None
        actual = greedy_text_generator.run(70, 'This')
        expected = 'This.'
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_empty_next_candidate(self):
        """
        Checks GreedyTextGenerator run method with empty next candidate
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._model.generate_next_token = lambda sequence: {}
        actual = greedy_text_generator.run(70, 'This')
        expected = 'This.'
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_invalid_input(self):
        """
        Checks GreedyTextGenerator run method with invalid inputs
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        expected = None
        seq_len_bad_input = [[None], {}, None, (), 1.1, 'string']
        prompt_bad_input = [1, [None], {}, None, (), 1.1, '', 0]
        for bad_input in seq_len_bad_input:
            actual = greedy_text_generator.run(bad_input, 'This')
            self.assertEqual(expected, actual)

        for bad_input in prompt_bad_input:
            actual = greedy_text_generator.run(70, bad_input)
            self.assertEqual(expected, actual)
