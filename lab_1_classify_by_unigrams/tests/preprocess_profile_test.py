# pylint: skip-file
"""
Checks the first lab profile preprocessing functions
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_1_classify_by_unigrams.main import preprocess_profile


class PreprocessProfileTest(unittest.TestCase):
    """
    Tests profile preprocessing function
    """

    PATH_TO_PROFILES_FOLDER = Path(__file__).parent.parent / 'assets' / 'profiles'

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_preprocess_profile_ideal(self):
        """
        Ideal scenario
        """

        expected = {
            'name': 'de',
            'freq': {
                'd': 0.0428, 'e': 0.1452, 'f': 0.0184, 'g': 0.0279, 'a': 0.0652, 'b': 0.0229,
                'c': 0.0356, 'l': 0.039, 'm': 0.0313, 'n': 0.0911, 'o': 0.0321, 'h': 0.0511,
                'i': 0.0792, 'j': 0.0037, 'k': 0.0166, 'u': 0.0384, 't': 0.0636, 'w': 0.0174,
                'v': 0.0081, 'q': 0.0002, 'p': 0.0113, 's': 0.0629, 'r': 0.0665, 'y': 0.0022,
                'x': 0.0015, 'z': 0.0103, '²': 1.1051e-05, '´': 3.8067e-05, 'ä': 0.0038, 'ü': 0.006,
                'ß': 0.0014, 'ö': 0.0024, 'é': 3.2337e-05, '̶': 2.4559e-05, '€': 0.0001,
                '▸': 3.9705e-05, '?': 1.187e-05}}

        path_to_profile = PreprocessProfileTest.PATH_TO_PROFILES_FOLDER / 'de.json'

        with open(path_to_profile, 'r', encoding='utf-8') as file:
            profile = json.load(file)

        actual = preprocess_profile(profile)
        self.assertEqual(expected['name'], actual['name'])

        for tuple_with_frequencies in expected['freq'].items():
            frequencies = actual['freq'][tuple_with_frequencies[0]]
            self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_preprocess_profile_bad_input_type(self):
        """
        Bad input scenario
        """
        expected = None

        profile = []
        actual = preprocess_profile(profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_preprocess_profile_bad_input_profile(self):
        """
        Bad input scenario
        """

        profile = {"n_words": [2179270, 2708449, 2118130], "name": "en"}

        expected = None

        actual = preprocess_profile(profile)
        self.assertEqual(expected, actual)
