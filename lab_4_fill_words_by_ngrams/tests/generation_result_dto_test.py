"""
GenerationResultDTO class tests
"""

import unittest

import pytest

from lab_4_fill_words_by_ngrams.main import GenerationResultDTO, GeneratorTypes


class GenerationResultDTOTest(unittest.TestCase):
    """
    Tests GenerationResultDTO class functionality
    """

    def setUp(self) -> None:
        self.text = 'Dinner and so the plate.'
        self.perplexity = 2.0476725110792193
        self.generator_type = GeneratorTypes()
        self.generators_types = [self.generator_type.greedy, self.generator_type.top_p,
                                 self.generator_type.beam_search]

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generation_result_dto_fields(self):
        """
        Checks if GenerationResultDTO fields are created correctly
        """
        generation_result_dto = GenerationResultDTO(self.text,
                                                    self.perplexity, 1)
        self.assertEqual(self.text, generation_result_dto.get_text())
        self.assertEqual(self.perplexity, generation_result_dto.get_perplexity())
        self.assertEqual(1, generation_result_dto.get_type())

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generation_result_dto_get_type(self):
        """
        Checks GenerationResultDTO get_type method
        ideal scenario
        """
        expected = [0, 1, 2]
        for index, generator_type in enumerate(self.generators_types):
            generator_result_dto = GenerationResultDTO(self.text,
                                                       self.perplexity,
                                                       generator_type)
            actual = generator_result_dto.get_type()
            self.assertEqual(expected[index], actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generation_result_dto_get_perplexity(self):
        """
        Checks GenerationResultDTO get_perplexity method ideal scenario
        """
        generator_result_dto = GenerationResultDTO(self.text,
                                                   self.perplexity,
                                                   self.generators_types[0])

        expected = self.perplexity
        actual = generator_result_dto.get_perplexity()
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generation_result_dto_get_text(self):
        """
        Checks GenerationResultDTO get_text method ideal scenario
        """
        generator_result_dto = GenerationResultDTO(self.text,
                                                   self.perplexity,
                                                   self.generators_types[0])

        expected = self.text
        actual = generator_result_dto.get_text()
        self.assertEqual(expected, actual)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generation_result_dto_get_result_in_str_format(self):
        """
        Checks GenerationResultDTO __str__ method
        """
        for generator_type in self.generators_types:
            generator_result_dto = GenerationResultDTO(self.text,
                                                       self.perplexity,
                                                       generator_type)
            actual = str(generator_result_dto)

            expected = (f'Perplexity score: {self.perplexity}\n'
                        f'{self.generator_type.get_conversion_generator_type(generator_type)}\n'
                        f'Text: {self.text}\n')

            self.assertEqual(expected, actual)
