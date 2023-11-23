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
