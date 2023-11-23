"""
GeneratorTypes class tests
"""

import unittest

import pytest

from lab_4_fill_words_by_ngrams.main import GeneratorTypes


class GeneratorTypesTest(unittest.TestCase):
    """
    Tests GeneratorTypes class functionality
    """

    def setUp(self) -> None:
        self.generator_types = GeneratorTypes()
        self.types = [self.generator_types.greedy, self.generator_types.top_p,
                      self.generator_types.beam_search]

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generator_types_fields(self):
        """
        Checks if GeneratorTypes fields are created correctly
        """
        generator_types = GeneratorTypes()
        self.assertEqual(generator_types.greedy, 0)
        self.assertEqual(generator_types.top_p, 1)
        self.assertEqual(generator_types.beam_search, 2)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_top_p_generator_run(self):
        """
        Checks GeneratorTypes get_conversion_generator_type method ideal scenario
        """
        expected = ['Greedy Generator', 'Top-P Generator', 'Beam Search Generator']
        for index, generator_type in enumerate(self.types):
            actual = self.generator_types.get_conversion_generator_type(generator_type)
            self.assertEqual(actual, expected[index])
