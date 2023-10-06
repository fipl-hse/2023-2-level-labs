"""
Checks the second lab's geo mean function
"""
import unittest

import pytest

from lab_2_tokenize_by_bpe.main import geo_mean


class GeoMeanTest(unittest.TestCase):
    """
    Tests geo mean function
    """
    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_geo_mean_ideal(self):
        """
        Ideal geo mean scenario
        """
        expected = 0.6878257000127244
        actual = geo_mean([0.8181818181818182, 0.6363636363636364, 0.625], 3)
        self.assertAlmostEqual(expected, actual, places=3)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_geo_mean_negative_value(self):
        """
        Geo mean negative precisions check
        """
        expected = 0.0
        actual = geo_mean([-1, 0, 0.5], 3)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_geo_mean_bad_input(self):
        """
        Geo mean invalid inputs check
        """
        precisions_bad_inputs = [(), 'string', {}, None, 1, 1.1, True]
        max_order_bad_inputs = [None, (), 1.1, [None], 'string', {}]
        expected = None
        for precision_bad_input in precisions_bad_inputs:
            actual = geo_mean(precision_bad_input, 3)
            self.assertEqual(expected, actual)
        for max_order_bad_input in max_order_bad_inputs:
            actual = geo_mean([-1, 0, 0.5], max_order_bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark10
    def test_geo_mean_return_value(self):
        """
        Geo mean return value check
        """
        actual = geo_mean([0.8181818181818182, 0.6363636363636364, 0.625], 3)
        self.assertTrue(isinstance(actual, float))
