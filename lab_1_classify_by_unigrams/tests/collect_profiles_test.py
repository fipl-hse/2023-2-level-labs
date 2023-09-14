"""
Checks the first lab language profile collection function
"""

import unittest
from pathlib import Path

import pytest

from lab_1_classify_by_unigrams.main import collect_profiles


class CollectProfilesTest(unittest.TestCase):
    """
    Tests profile collection function
    """

    PATH_TO_PROFILES_FOLDER = Path(__file__).parent.parent / 'assets' / 'profiles'

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_collect_profiles_ideal(self):
        """
        Ideal scenario
        """

        expected = [
            {'name': 'it', 'freq':
                {'d': 0.0339, 'e': 0.1039, 'f': 0.0118, 'g': 0.0183, 'a': 0.1132, 'b': 0.012,
                 'c': 0.046, 'l': 0.0548, 'm': 0.0322, 'n': 0.07, 'o': 0.1004, 'h': 0.017,
                 'i': 0.1063, 'j': 0.0007, 'k': 0.0019, 'u': 0.0297, 't': 0.0619, 'w': 0.002,
                 'v': 0.0176, 'q': 0.0044, 'p': 0.0293, 's': 0.0508, 'r': 0.0602, 'y': 0.0014,
                 'x': 0.0013, 'z': 0.0081, 'è': 0.004, 'í': 1.3936e-05, 'ì': 0.0008, 'é': 0.0004,
                 'ç': 0.0004, 'á': 1.8291e-05, 'à': 0.0015, 'ú': 1.0016e-05, 'ù': 0.001,
                 'ò': 0.0009, 'ó': 3.5711e-05, '̸': 1.0452e-05, '€': 0.0001, '?': 1.5678e-05}},
            {'name': 'ru', 'freq':
                {'©': 2.7482e-05, 'ь': 0.0193, 'э': 0.003, 'ю': 0.0064, 'я': 0.0215, 'ш': 0.0077,
                 'щ': 0.0031, 'ъ': 0.0002, 'ы': 0.0171, 'ф': 0.004, 'х': 0.0095, 'ц': 0.0045,
                 'ч': 0.0148, 'р': 0.0499, 'с': 0.0523, 'т': 0.0659, 'у': 0.0287, 'ё': 0.0008,
                 'й': 0.0124, 'и': 0.0705, 'л': 0.0398, 'к': 0.0422, 'н': 0.0629, 'м': 0.0306,
                 'п': 0.0283, 'о': 0.1, 'б': 0.0188, 'а': 0.0854, 'г': 0.016, 'в': 0.0418,
                 'е': 0.0828, 'д': 0.0313, 'з': 0.0166, 'ж': 0.0096, '√': 1.5704e-05,
                 '№': 1.5704e-05, '😂': 3.5334e-05, '😘': 1.963e-05, '🙋': 2.5519e-05,
                 '👍': 1.5704e-05, '가': 2.3556e-05, '?': 0.0003}},
            {'name': 'tr', 'freq':
                {'d': 0.0434, 'e': 0.0928, 'f': 0.0066, 'g': 0.0155, 'a': 0.1211, 'b': 0.0282,
                 'c': 0.0129, 'l': 0.0575, 'm': 0.047, 'n': 0.0705, 'o': 0.0325, 'h': 0.0136,
                 'i': 0.0868, 'j': 0.0007, 'k': 0.0441, 'u': 0.0342, 't': 0.0331, 'w': 0.0009,
                 'v': 0.0096, 'q': 0.0001, 'p': 0.0093, 's': 0.038, 'r': 0.062, 'y': 0.038,
                 'x': 0.0001, 'z': 0.0176, 'ç': 0.008, 'ü': 0.0126, 'ß': 1.3317e-05, 'ö': 0.0052,
                 'î': 1.645e-05, 'é': 1.0183e-05, 'â': 7.8728e-05, 'û': 1.4492e-05, 'ğ': 0.0062,
                 'ı': 0.0371, 'i̇': 0.001, 'ş': 0.0122, '̅': 1.175e-05, '█': 1.7234e-05,
                 '\ue022': 3.956e-05}}]

        paths_to_profiles = [
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'it.json'),
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'ru.json'),
            str(CollectProfilesTest.PATH_TO_PROFILES_FOLDER / 'tr.json')]

        actual = collect_profiles(paths_to_profiles)
        for expected_dictionary_with_profiles, actual_dictionary_with_profiles \
                in zip(expected, actual):
            self.assertEqual(expected_dictionary_with_profiles['name'],
                             actual_dictionary_with_profiles['name'])

            for tuple_with_frequencies in expected_dictionary_with_profiles['freq'].items():
                frequencies = actual_dictionary_with_profiles['freq'][tuple_with_frequencies[0]]
                self.assertAlmostEqual(tuple_with_frequencies[1], frequencies, delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark10
    def test_preprocess_profile_bad_input_type(self):
        """
        Bad input scenario
        """
        expected = None

        bad_inputs = ['string', {}, (), None, 9, 9.34, True]
        for bad_input in bad_inputs:
            actual = collect_profiles(bad_input)
            self.assertEqual(expected, actual)
