"""
Checks the first lab language detection function
"""

import unittest

import pytest

from lab_1_classify_by_unigrams.main import detect_language_advanced


class DetectLanguageAdvancedTest(unittest.TestCase):
    """
    Tests language detection advanced function
    """

    known_profile = [
        {'name': 'de', 'freq':
            {'d': 0.0428, 'e': 0.1452, 'f': 0.0184, 'g': 0.0279, 'a': 0.0652, 'b': 0.0229,
             'c': 0.0356, 'l': 0.039, 'm': 0.0313, 'n': 0.0911, 'o': 0.0321, 'h': 0.0511,
             'i': 0.0792, 'j': 0.0037, 'k': 0.0166, 'u': 0.0384, 't': 0.0636, 'w': 0.0174,
             'v': 0.0081, 'q': 0.0002, 'p': 0.0113, 's': 0.0629, 'r': 0.0665, 'y': 0.0022,
             'x': 0.0015, 'z': 0.0103, '²': 1.1051e-05, '´': 3.8067e-05, 'ä': 0.0038, 'ü': 0.006,
             'ß': 0.0014, 'ö': 0.0024, 'é': 3.2337e-05, '̶': 2.4559e-05, '€': 0.0001,
             '▸': 3.9705e-05, '?': 1.187e-05}},
        {'name': 'en', 'freq':
            {'d': 0.0352, 'e': 0.1106, 'f': 0.02, 'g': 0.0247, 'a': 0.0788, 'b': 0.0177,
             'c': 0.0252, 'l': 0.0454, 'm': 0.0292, 'n': 0.066, 'o': 0.087, 'h': 0.0476,
             'i': 0.0724, 'j': 0.0032, 'k': 0.0135, 'u': 0.0324, 't': 0.0855, 'w': 0.0249,
             'v': 0.0111, 'q': 0.0006, 'p': 0.0186, 's': 0.0612, 'r': 0.0542, 'y': 0.0287,
             'x': 0.0036, 'z': 0.0013, '£': 8.1678e-05, '´': 9.0856e-05, 'é': 2.2484e-05,
             'ä': 1.2848e-05, 'á': 1.0553e-05, 'ü': 2.8449e-05, 'ö': 1.0095e-05, '̶': 2.2484e-05,
             '̩': 1.3766e-05, '€': 1.3307e-05, '▸': 3.625e-05}},
        {'name': 'es', 'freq':
            {'d': 0.0453, 'e': 0.1311, 'f': 0.0069, 'g': 0.0132, 'a': 0.125, 'b': 0.0133,
             'c': 0.0377, 'l': 0.0511, 'm': 0.032, 'n': 0.0651, 'o': 0.0904, 'h': 0.0127,
             'i': 0.0554, 'j': 0.0092, 'k': 0.0015, 'u': 0.041, 't': 0.0435, 'w': 0.0011,
             'v': 0.0125, 'q': 0.0132, 'p': 0.0272, 's': 0.0716, 'r': 0.0608, 'y': 0.013,
             'x': 0.0027, 'z': 0.0031, 'ª': 3.3602e-05, '¬': 0.0001, '¡': 0.0005, 'º': 5.9781e-05,
             '¿': 0.0009, '·': 1.3284e-05, '´': 1.9927e-05, 'é': 0.0025, 'á': 0.0035, 'ú': 0.0009,
             'ñ': 0.0023, 'í': 0.0044, 'è': 3.3993e-05, 'ç': 0.0001, 'à': 9.6119e-05,
             'ü': 5.392e-05, 'ò': 4.5324e-05, 'ó': 0.003, '̶': 1.8364e-05, '€': 7.5411e-05,
             '▸': 3.0867e-05}},
        {'name': 'fr', 'freq':
            {'d': 0.0339, 'e': 0.1423, 'f': 0.0122, 'g': 0.0099, 'a': 0.0822, 'b': 0.0115,
             'c': 0.0333, 'l': 0.0494, 'm': 0.0329, 'n': 0.0648, 'o': 0.0611, 'h': 0.0122,
             'i': 0.071, 'j': 0.0123, 'k': 0.0023, 'u': 0.0619, 't': 0.065, 'w': 0.0021,
             'v': 0.0167, 'q': 0.0103, 'p': 0.0312, 's': 0.0788, 'r': 0.0649, 'y': 0.0043,
             'x': 0.0045, 'z': 0.0022, '²': 1.0321e-05, 'é': 0.0141, 'ê': 0.0017, 'ç': 0.0018,
             'à': 0.0039, 'ï': 5.5645e-05, 'î': 0.0001, 'ë': 2.4681e-05, 'è': 0.0024, 'â': 0.0002,
             'û': 0.0001, 'ù': 0.0001, 'ô': 0.0004, 'œ': 4.0388e-05, '€': 0.0001, '→': 1.1667e-05,
             '▸': 3.4105e-05}}]

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_advanced_ideal(self):
        """
        Ideal scenario
        """

        unknown_profile = {
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538,
                'm': 0.0769, 'i': 0.0769, 'a': 0.2307, 's': 0.0769,
                'p': 0.1538
            }
        }

        expected = [('es', 0.0016), ('fr', 0.0021),
                    ('en', 0.0022), ('de', 0.0024)]
        actual = detect_language_advanced(unknown_profile, DetectLanguageAdvancedTest.known_profile)

        for expected_tuple_with_distance, actual_tuple_with_distance in zip(expected, actual):
            self.assertAlmostEqual(expected_tuple_with_distance[1], actual_tuple_with_distance[1],
                                   delta=1e-4)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_advanced_bad_input_unknown_profile(self):
        """
        Bad input scenario
        """

        unknown_profile = ''

        expected = None
        actual = detect_language_advanced(unknown_profile, DetectLanguageAdvancedTest.known_profile)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_classify_by_unigrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_detect_language_advanced_bad_input_known_profile(self):
        """
        Bad input scenario
        """

        unknown_profile = {
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538,
                'm': 0.0769, 'i': 0.0769, 'a': 0.2307, 's': 0.0769,
                'p': 0.1538
            }
        }

        known_profile = {}

        expected = None
        actual = detect_language_advanced(unknown_profile, known_profile)
        self.assertEqual(expected, actual)
