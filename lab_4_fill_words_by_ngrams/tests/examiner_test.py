# pylint: disable=protected-access, duplicate-code, unused-argument
"""
Examiner class tests
"""
import unittest
from unittest.mock import mock_open, patch

import pytest

from config.constants import PROJECT_ROOT
from lab_3_generate_by_ngrams.main import NGramLanguageModel
from lab_4_fill_words_by_ngrams.main import (Examiner, GeneratorRuleStudent, GeneratorTypes,
                                             WordProcessor)


class ExaminerTest(unittest.TestCase):
    """
    Tests Examiner class functionality
    """

    def setUp(self) -> None:
        text = '''"That's your problem, isn't it?" said Filch, his voice cracking with
        glee. "Should've thought of them werewolves before you got in trouble,
        shouldn't you?"
        Hagrid came striding toward them out of the dark, Fang at his heel. He
        was carrying his large crossbow, and a quiver of arrows hung over his
        shoulder.
        "Abou' time," he said. "I bin waitin' fer half an hour already. All
        right, Harry, Hermione?"'''
        self.json_path = str(PROJECT_ROOT / 'lab_4_fill_words_by_ngrams' / 'assets' /
                             'question_and_answers.json')
        self.word_processor = WordProcessor('<eos>')
        self.encoded = self.word_processor.encode(text)
        self.language_model = NGramLanguageModel(self.encoded, 2)
        self.language_model.build()
        generators_types = GeneratorTypes()
        greedy_student = GeneratorRuleStudent(generators_types.greedy, self.language_model,
                                              self.word_processor)
        beam_search_student = GeneratorRuleStudent(
            generators_types.beam_search, self.language_model, self.word_processor
        )
        top_p_student = GeneratorRuleStudent(generators_types.top_p, self.language_model,
                                             self.word_processor)
        self.generators = [greedy_student, beam_search_student, top_p_student]
        self.examiner = Examiner(self.json_path)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_fields(self):
        """
        Checks if Examiner fields are created correctly
        """
        self.assertEqual(self.json_path, self.examiner._json_path)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_load_from_json(self):
        """
        Checks Examiner _load_from_json method ideal scenario
        """
        expected = {
            ("A sign overhead said Express eleven o'clock.", 21):
                "A sign overhead said Hogwarts Express eleven o'clock.",
            ('Professor rolled up her scroll and took the sorting hat away.', 10):
                'Professor mcgonagall rolled up her scroll and took the sorting hat away.',
            ('Hagrid came striding toward them out the dark fang at his heel.', 37):
                'Hagrid came striding toward them out of the dark fang at his heel.',
            ('In danger of being speared on the end of an umbrella by a bearded giant '
             'uncle courage failed again.', 78):
                'In danger of being speared on the end of an umbrella by a bearded giant '
                'uncle vernon courage failed again.', ('Gryffindor will in even more trouble.', 16):
                'Gryffindor will be in even more trouble.',
            ('Soon there a huddle of limp black players slumped along the wall.', 11):
                'Soon there was a huddle of limp black players slumped along the wall.',
            ('I m holding the house cup and the cup.', 22): 'I m holding the house cup and the '
                                                            'quidditch cup',
            ('Wingardium is a levitation charm.', 11): 'Wingardium leviosa is a levitation charm.',
            ('Harry looked behind him and saw a wrought iron archway where the barrier had been '
             'with the words platform and three quarters on it.', 106):
                'Harry looked behind him and saw a wrought iron archway where the barrier had '
                'been with the words platform nine and three quarters on it.',
            ('Albus is the headmaster of the wizarding school hogwarts.', 6):
                'Albus dumbledore is the headmaster of the wizarding school hogwarts.'}
        actual = self.examiner._load_from_json()
        self.assertEqual(actual, expected)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_load_from_json_invalid_input(self):
        """
        Checks Examiner _load_from_json method with invalid inputs
        """
        bad_inputs = [[None], 1, None, (), 1.1, {}]

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                Examiner(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_load_from_json_empty_input(self):
        """
        Checks Examiner _load_from_json method with empty input
        """
        bad_input = ''

        with self.assertRaises(ValueError):
            Examiner(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_load_from_json_invalid_file_extension(self):
        """
        Checks Examiner _load_from_json method with inappropriate
        file extension
        """
        self.examiner._json_path = str(PROJECT_ROOT / 'lab_4_fill_words_by_ngrams' / 'assets' /
                                       'Harry_Potter.txt')
        with self.assertRaises(ValueError):
            self.examiner._load_from_json()

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    @patch('builtins.open', new_callable=mock_open,
           read_data='''{
           "question": "A sign overhead said Express eleven o'clock.", 
           "location": 21,
           "answer": "A sign overhead said Hogwarts Express eleven o'clock."}''')
    def test_examiner_load_from_json_invalid_data_from_json(self, mocked_file):
        """
        Checks Examiner _load_from_json method with invalid data from json
        """
        self.examiner._json_path = 'Harry_Potter.json'
        with self.assertRaises(ValueError):
            self.examiner._load_from_json()

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_provide_questions(self):
        """
        Checks Examiner provide_questions method ideal scenario
        """
        examiner = Examiner(self.json_path)
        expected = [
            ("A sign overhead said Express eleven o'clock.", 21),
            ('Professor rolled up her scroll and took the sorting hat away.', 10),
            ('Hagrid came striding toward them out the dark fang at his heel.', 37),
            ('In danger of being speared on the end of an umbrella by a bearded giant uncle '
             'courage failed again.', 78),
            ('Gryffindor will in even more trouble.', 16),
            ('Soon there a huddle of limp black players slumped along the wall.', 11),
            ('I m holding the house cup and the cup.', 22),
            ('Wingardium is a levitation charm.', 11),
            ('Harry looked behind him and saw a wrought iron archway where the barrier had '
             'been with the words platform and three quarters on it.', 106),
            ('Albus is the headmaster of the wizarding school hogwarts.', 6)]
        actual = examiner.provide_questions()
        self.assertEqual(actual, expected)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_assess_exam(self):
        """
        Checks Examiner provide_questions method ideal scenario
        """
        examiner = Examiner(self.json_path)
        expected = [0.1, 0.1, 0.1]
        for index, student in enumerate(self.generators):
            student_answers = student.take_exam(examiner.provide_questions())
            actual = examiner.assess_exam(student_answers)
            self.assertEqual(actual, expected[index])

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_assess_exam_invalid_input(self):
        """
        Checks Examiner provide_questions method with invalid inputs
        """
        examiner = Examiner(self.json_path)
        bad_inputs = [[None], 1, None, (), 1.1, 'string']

        with self.assertRaises(ValueError):
            for bad_input in bad_inputs:
                examiner.assess_exam(bad_input)

    @pytest.mark.lab_4_fill_words_by_ngrams
    @pytest.mark.mark10
    def test_examiner_assess_exam_empty_input(self):
        """
        Checks Examiner provide_questions method with invalid inputs
        """
        examiner = Examiner(self.json_path)
        bad_input = {}

        with self.assertRaises(ValueError):
            examiner.assess_exam(bad_input)
