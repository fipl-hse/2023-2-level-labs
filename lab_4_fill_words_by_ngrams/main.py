"""
Lab 4.

Top-p sampling generation and filling gaps with ngrams
"""
import json
import math
from random import choice

# pylint:disable=too-few-public-methods, too-many-arguments
from lab_3_generate_by_ngrams.main import (BeamSearchTextGenerator, GreedyTextGenerator,
                                           NGramLanguageModel, TextProcessor)


class WordProcessor(TextProcessor):
    """
    Handle text tokenization, encoding and decoding.
    """

    def _tokenize(self, text: str) -> tuple[str, ...]:  # type: ignore
        """
        Tokenize text into tokens, separating sentences with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in sentence.

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...]: Tokenized text

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(text, str) or not text:
            raise ValueError('Type input is inappropriate or input argument is empty.')
        text = repr(text)
        text = text.replace('\\n', ' ')
        preprocessed_text = ''
        for element in text.lower():
            if element in '?!.':
                preprocessed_text += ' '
                preprocessed_text += self.get_end_of_word_token()
            elif element.isalpha() or element.isspace():
                preprocessed_text += element

        return tuple(preprocessed_text.split(' '))

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): Element to put into storage

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(element, str):
            raise ValueError('Type input is inappropriate.')
        if not element:
            raise ValueError('Input argument is empty.')
        if element not in self._storage:
            self._storage[element] = len(self._storage)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str:  # type: ignore
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces.
        The first word is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): Tuple of decoded tokens

        Returns:
            str: Resulting text

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            raise ValueError('Type input is inappropriate or input argument is empty.')
        result = ''
        for word in decoded_corpus:
            if word == self.get_end_of_word_token():
                result = result[:-1] + '.'
            else:
                for letter in word:
                    if not result or (len(result) > 2 and result[-2] == '.'):
                        result += letter.upper()
                    else:
                        result += letter
            result += ' '
        result = result.strip()
        if result[-1] != '.':
            result += '.'
        return result


class TopPGenerator:
    """
    Generator with top-p sampling strategy.

    Attributes:
        _model (NGramLanguageModel): NGramLanguageModel instance to use for text generation
        _word_processor (WordProcessor): WordProcessor instance to handle text processing
        _p_value (float) : Collective probability mass threshold for generation
    """

    def __init__(
            self, language_model: NGramLanguageModel, word_processor: WordProcessor, p_value: float
    ) -> None:
        """
        Initialize an instance of TopPGenerator.

        Args:
            language_model (NGramLanguageModel):
                NGramLanguageModel instance to use for text generation
            word_processor (WordProcessor): WordProcessor instance to handle text processing
            p_value (float): Collective probability mass threshold
        """
        self._model = language_model
        self._word_processor = word_processor
        self._p_value = p_value

    def run(self, seq_len: int, prompt: str) -> str:  # type: ignore
        """
        Generate sequence with top-p sampling strategy.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of the sequence

        Returns:
            str: Generated sequence

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if sequence has inappropriate length,
                or if methods used return None.
        """
        if not isinstance(seq_len, int) or seq_len < 0 \
                or not isinstance(prompt, str) or not prompt:
            raise ValueError('Type input is inappropriate or input argument is empty.')
        encoded_prompt = self._word_processor.encode(prompt)
        if encoded_prompt is None:
            raise ValueError('None is returned')
        encoded_list = list(encoded_prompt)
        for _ in range(seq_len):
            candidates = self._model.generate_next_token(encoded_prompt)
            if candidates is None:
                raise ValueError('None is returned')
            if not candidates:
                break
            tuple_candidates = tuple(candidates.items())
            sorted_candidates = sorted(tuple_candidates, key=lambda tup: (-tup[1], -tup[0]))
            sum_freq = 0
            num_candidates = 0
            for candidate in sorted_candidates:
                if sum_freq >= self._p_value:
                    break
                sum_freq += candidate[1]
                num_candidates += 1
            rand_token = choice(sorted_candidates[:num_candidates])[0]
            encoded_list.append(rand_token)
            encoded_prompt = tuple(encoded_list)
        decoded = self._word_processor.decode(encoded_prompt)
        if decoded is None:
            raise ValueError('None is returned')
        return decoded


class GeneratorTypes:
    """
    A class that represents types of generators.

    Attributes:
        greedy (int): Numeric type of Greedy generator
        top_p (int): Numeric type of Top-P generator
        beam_search (int): Numeric type of Beam Search generator
    """

    def __init__(self) -> None:
        """
        Initialize an instance of GeneratorTypes.
        """
        self.greedy = 0
        self.top_p = 1
        self.beam_search = 2

    def get_conversion_generator_type(self, generator_type: int) -> str:  # type: ignore
        """
        Retrieve string type of generator.

        Args:
            generator_type (int): Numeric type of the generator

        Returns:
            (str): Name of the generator.
        """
        if generator_type == self.greedy:
            return 'greedy generator'
        if generator_type == self.top_p:
            return 'top-p generator'
        if generator_type == self.beam_search:
            return 'beam search generator'
        return ''


class GenerationResultDTO:
    """
    Class that represents results of QualityChecker.

    Attributes:
        __text (str): Text that used to calculate perplexity
        __perplexity (float): Calculated perplexity score
        __type (int): Numeric type of the generator
    """

    def __init__(self, text: str, perplexity: float, generation_type: int):
        """
        Initialize an instance of GenerationResultDTO.

        Args:
            text (str): The text used to calculate perplexity
            perplexity (float): Calculated perplexity score
            generation_type (int):
                Numeric type of the generator for which perplexity was calculated
        """
        self.__text = text
        self.__perplexity = perplexity
        self.__type = generation_type

    def get_perplexity(self) -> float:  # type: ignore
        """
        Retrieve a perplexity value.

        Returns:
            (float): Perplexity value
        """
        return self.__perplexity

    def get_text(self) -> str:  # type: ignore
        """
        Retrieve a text.

        Returns:
            (str): Text for which the perplexity was count
        """
        return self.__text

    def get_type(self) -> int:  # type: ignore
        """
        Retrieve a numeric type.

        Returns:
            (int): Numeric type of the generator
        """
        return self.__type

    def __str__(self) -> str:  # type: ignore
        """
        Prints report after quality check.

        Returns:
            (str): String with report
        """
        generator_types = GeneratorTypes()
        return (f'Perplexity score: {self.__perplexity}\n'
                f'{generator_types.get_conversion_generator_type(self.__type)}\n'
                f'Text: {self.__text}\n')


class QualityChecker:
    """
    Check the quality of different ways to generate sequence.

    Attributes:
        _generators (dict): Dictionary with generators to check quality
        _language_model (NGramLanguageModel): NGramLanguageModel instance
        _word_processor (WordProcessor): WordProcessor instance
    """

    def __init__(
        self, generators: dict, language_model: NGramLanguageModel, word_processor: WordProcessor
    ) -> None:
        """
        Initialize an instance of QualityChecker.

        Args:
            generators (dict): Dictionary in the form of {numeric type: generator}
            language_model (NGramLanguageModel):
                NGramLanguageModel instance to use for text generation
            word_processor (WordProcessor): WordProcessor instance to handle text processing
        """
        self._generators = generators
        self._language_model = language_model
        self._word_processor = word_processor

    def _calculate_perplexity(self, generated_text: str) -> float:  # type: ignore
        """
        Calculate perplexity for the text made by generator.

        Args:
            generated_text (str): Text made by generator

        Returns:
            float: Perplexity score for generated text

        Raises:
            ValueError: In case of inappropriate type input argument,
                or if input argument is empty,
                or if methods used return None,
                or if nothing was generated.
        """
        if not generated_text:
            raise ValueError('Input argument is empty')
        if not isinstance(generated_text, str):
            raise ValueError('Inappropriate type argument')
        encoded_text = self._word_processor.encode(generated_text)
        if encoded_text is None:
            raise ValueError('self._word_processor.encode() returned None')
        l_sum = 0.0
        num_ngrams = 0.0
        ngrams = self._language_model.generate_next_token(encoded_text)
        if ngrams is None:
            raise ValueError('self._language_model.generate_next_token() returned None')
        for ngram in ngrams:
            l_sum += math.log(ngrams[ngram])
            num_ngrams += 1
        l_sum = -(l_sum / num_ngrams)
        result = pow(math.e, l_sum)
        if type(result) == float:
            return result
        return 0.0

    def run(self, seq_len: int, prompt: str) -> list[GenerationResultDTO]:  # type: ignore
        """
        Check the quality of generators.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of the sequence

        Returns:
            list[GenerationResultDTO]:
                List of GenerationResultDTO instances in ascending order of perplexity score

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if sequence has inappropriate length,
                or if methods used return None.
        """
        if not seq_len:
            raise ValueError('Input argument seq_len is empty')
        if not prompt:
            raise ValueError('Input argument prompt is empty')
        if not isinstance(seq_len, int):
            raise ValueError('Inappropriate type argument seq_len')
        if not isinstance(prompt, str):
            raise ValueError('Inappropriate type argument prompt')

        generators_inv = {value: key for key, value in self._generators.items()}
        results_list = []

        for generator, num_type in generators_inv.items():
            text = generator.run(seq_len, prompt)
            if text is None:
                raise ValueError(f'{generator} methode run() returned None')
            perplexity = self._calculate_perplexity(text)
            if perplexity is None:
                raise ValueError(f'{generator} perplexity is None')
            result = GenerationResultDTO(text, perplexity, num_type)
            results_list.append((result, result.get_perplexity(), result.get_type()))

        sorted_results = sorted(results_list, key=lambda tup: (tup[2], tup[1]))

        return [res_tuple[0] for res_tuple in sorted_results]


class Examiner:
    """
    A class that conducts an exam.

    Attributes:
        _json_path (str): Local path to assets file
        _questions_and_answers (dict[tuple[str, int], str]):
            Dictionary in the form of {(question, position of the word to be filled), answer}
    """

    def __init__(self, json_path: str) -> None:
        """
        Initialize an instance of Examiner.

        Args:
            json_path (str): Local path to assets file
        """
        if not isinstance(json_path, str) or not json_path:
            raise ValueError
        self._json_path = json_path
        self._questions_and_answers = {}

    def _load_from_json(self) -> dict[tuple[str, int], str]:  # type: ignore
        """
        Load questions and answers from JSON file.

        Returns:
            dict[tuple[str, int], str]:
                Dictionary in the form of {(question, position of the word to be filled), answer}

        Raises:
            ValueError: In case of inappropriate type of attribute _json_path,
                or if attribute _json_path is empty,
                or if attribute _json_path has inappropriate extension,
                or if inappropriate type loaded data.
        """
        if not isinstance(self._json_path, str):
            raise ValueError('Inappropriate type of attribute _json_path')
        if not self._json_path:
            raise ValueError('Attribute _json_path is empty')
        if not self._json_path.endswith('json'):
            raise ValueError('Attribute _json_path has inappropriate extension')
        with open(self._json_path, 'r', encoding="utf-8") as file:
            self._questions_and_answers = json.load(file)
        if not isinstance(self._questions_and_answers, list):
            raise ValueError('Inappropriate type loaded data')

    def provide_questions(self) -> list[tuple[str, int]]:  # type: ignore
        """
        Provide questions for an exam.

        Returns:
            list[tuple[str, int]]:
                List in the form of [(question, position of the word to be filled)]
        """
        questions = [(question, place) for (question, place), answer
                     in self._questions_and_answers.items()]
        return questions

    def assess_exam(self, answers: dict[str, str]) -> float:  # type: ignore
        """
        Assess an exam by counting accuracy.

        Args:
            answers(dict[str, str]): Dictionary in the form of {question: answer}

        Returns:
            float: Accuracy score

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(answers, dict):
            raise ValueError('Inappropriate type input argument')
        if not answers:
            raise ValueError('Input argument is empty')
        num_questions = 0
        score = 0
        right_answers = {question: answer for (question, place), answer
                         in self._questions_and_answers.items()}
        for question in answers:
            num_questions += 1
            if answers[question] == right_answers[question]:
                score += 1
        return score / num_questions


class GeneratorRuleStudent:
    """
    Base class for students generators.
    """

    _generator: GreedyTextGenerator | TopPGenerator | BeamSearchTextGenerator
    _generator_type: int

    def __init__(
        self, generator_type: int, language_model: NGramLanguageModel, word_processor: WordProcessor
    ) -> None:
        """
        Initialize an instance of GeneratorRuleStudent.

        Args:
            generator_type (int): Numeric type of the generator
            language_model (NGramLanguageModel):
                NGramLanguageModel instance to use for text generation
            word_processor (WordProcessor): WordProcessor instance to handle text processing
        """
        self._generator_type = generator_type
        if generator_type == 0:
            self._generator = GreedyTextGenerator(language_model, word_processor)
        elif generator_type == 1:
            self._generator = TopPGenerator(language_model, word_processor, 0.5)
        elif self._generator_type == 2:
            self._generator = BeamSearchTextGenerator(language_model, word_processor, 5)

    def take_exam(self, tasks: list[tuple[str, int]]) -> dict[str, str]:  # type: ignore
        """
        Take an exam.

        Args:
            tasks (list[tuple[str, int]]):
                List with questions in the form of [(question, position of the word to be filled)]

        Returns:
            dict[str, str]: Dictionary in the form of {question: answer}

        Raises:
            ValueError: In case of inappropriate type input argument,
                or if input argument is empty,
                or if methods used return None.
        """
        if not isinstance(tasks, list):
            raise ValueError('Inappropriate type input argument')
        if not tasks:
            raise ValueError('Input argument is empty')
        answers = {}
        for (question, place) in tasks:
            context = question[:place]
            answer = self._generator.run(seq_len=1, prompt=context)
            if answer is None:
                raise ValueError('self._generator.run() returned None')
            if answer[-1] == '.':
                answer = answer[:-1]
            result = answer + question[place:]
            answers[question] = result
        return answers

    def get_generator_type(self) -> str:  # type: ignore
        """
        Retrieve generator type.

        Returns:
            str: Generator type
        """
        generator_types = GeneratorTypes()
        return generator_types.get_conversion_generator_type(self._generator_type)
