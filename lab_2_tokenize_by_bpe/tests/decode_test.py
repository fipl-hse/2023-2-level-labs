"""
Checks the second lab's decode function
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_2_tokenize_by_bpe.main import decode


class DecodeTest(unittest.TestCase):
    """
    Tests decoding function
    """

    def setUp(self) -> None:
        path_to_tests_directory = Path(__file__).parent
        with open(path_to_tests_directory / 'vocabulary.json', 'r', encoding='utf-8') as json_file:
            self.vocabulary = json.load(json_file)
        with open(path_to_tests_directory / 'encoded_text.json', 'r',
                  encoding='utf-8') as json_file:
            loaded_dict = json.load(json_file)
            self.encoded_ideal = loaded_dict["ideal_encoded_text"]
            self.encoded_with_unk = loaded_dict["encoded_text_with_unk"]
            self.encoded_without_end = loaded_dict["encoded_text_without_end"]

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_ideal(self):
        """
        Ideal decode scenario
        """
        expected = ('В поисках пищи альбатросы способны преодолевать '
                    'значительные расстояния при малой затрате сил, '
                    'используя наклонное либо динамическое парение. '
                    'Их крылья устроены так, что птица может долго '
                    'зависать в воздухе, но не осиливает длительный '
                    'маховый полет. Активный взмах крыльями альбатрос '
                    'делает только при взлете, полагаясь далее на '
                    'силу и направление ветра. ')

        actual = decode(self.encoded_ideal, self.vocabulary, '</s>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_with_unk(self):
        """
        Decode scenario with unknown tokens
        """
        expected = ('Это интересно! Слово «альбатрос» произошло '
                    'от арабского al-<unk>a<unk><unk><unk>s («ныряльщик»), '
                    'которое на португальском наречии стало звучать как '
                    'alcatra<unk>, перекочевав затем в английский и русский '
                    'языки. Под влиянием латинского al<unk><unk>s («белый») '
                    'alcatra<unk> чуть позднее превратился в al<unk>atross. '
                    'Алькатрас – так назван остров в Калифорнии, где '
                    'содержались особо опасные преступники. ')
        actual = decode(self.encoded_with_unk, self.vocabulary, '</s>')
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_without_end_token(self):
        """
        Decode scenario without end token
        """
        expected = ('Впоискахпищиальбатросыспособныпреодолеватьзначительныерасстояния'
                    'прималойзатратесил,используянаклонноелибодинамическоепарение.Их'
                    'крыльяустроенытак,чтоптицаможетдолгозависатьввоздухе,нонеосиливает'
                    'длительныймаховыйполет.Активныйвзмахкрыльямиальбатросделаеттолькопри'
                    'взлете,полагаясьдалеенасилуинаправлениеветра.')
        actual = decode(self.encoded_without_end, self.vocabulary, None)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_bad_input(self):
        """
        Decode invalid inputs check
        """
        encoded_text_bad_input = ['string', (), {}, None, 1, 1.1, True]
        vocabulary_bad_input = [None, (), 1.1, True, [None], 'string', 1]
        end_of_word_bad_input = [(), {}, 1, 1.1, True, [None]]
        expected = None
        for index, bad_input in enumerate(encoded_text_bad_input):
            actual = decode(bad_input, self.vocabulary, '</s>')
            self.assertEqual(expected, actual)

            actual = decode(self.encoded_ideal, vocabulary_bad_input[index], '</s>')
            self.assertEqual(expected, actual)

        for bad_input in end_of_word_bad_input:
            actual = decode(self.encoded_ideal, self.vocabulary, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_tokenize_by_bpe
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_return_value(self):
        """
        Decode return value check
        """
        actual = decode(self.encoded_ideal, self.vocabulary, '</s>')
        self.assertTrue(isinstance(actual, str))
