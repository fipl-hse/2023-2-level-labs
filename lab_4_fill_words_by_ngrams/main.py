"""
Lab 4.

Top-p sampling generation and filling gaps with ngrams
"""
#pylint:disable=too-few-public-methods, too-many-arguments

import math
from random import choice

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
        if not isinstance(text, str) or text is None or len(text) == 0:
            raise ValueError('Type input is inappropriate or input argument is empty.')
        tokenized_text = ''
        for word in text.lower():
            if word.isalpha() or word.isspace():
                tokenized_text += word
            elif word in ('?', '.', '!'):
                tokenized_text += f' {self._end_of_word_token}'
        return tuple(tokenized_text.split())

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): Element to put into storage

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(element, str) or element is None or len(element) == 0:
            raise ValueError('Type input is inappropriate or input argument is empty.')
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
        if (not isinstance(decoded_corpus, tuple) or decoded_corpus is None
                or len(decoded_corpus) == 0):
            raise ValueError('Type input is inappropriate or input argument is empty.')
        tokenized_text = list(decoded_corpus)
        if tokenized_text is None:
            raise ValueError('Type input is inappropriate or input argument is empty.')
        tokenized_text[0] = tokenized_text[0].capitalize()
        if self.get_end_of_word_token() in tokenized_text:
            for i in range(1, len(tokenized_text)):
                if tokenized_text[i - 1] == self.get_end_of_word_token():
                    tokenized_text[i] = tokenized_text[i].capitalize()
        new_text = ' '.join(tokenized_text).replace(f' {self._end_of_word_token}', '.')
        if new_text[-1] != '.':
            new_text += '.'
        return new_text


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
        if (not isinstance(seq_len, int) or seq_len <= 0 or not isinstance(prompt, str)
                or prompt is None):
            raise ValueError('Type input is inappropriate or input argument is empty.')
        encoded_text = self._word_processor.encode(prompt)
        if encoded_text is None:
            raise ValueError('Encoding is not working.')
        for i in range(seq_len):
            tokens = self._model.generate_next_token(encoded_text)
            if tokens is None:
                raise ValueError('None is returned.')
            if tokens is None:
                break
            sorted_tokens = sorted(tokens.items(), key=lambda x: (x[1], x[0]), reverse=True)
            list_of_tokens = []
            sum_freq = 0
            for token in sorted_tokens:
                sum_freq += token[1]
                list_of_tokens.append(token[0])
                if sum_freq >= self._p_value:
                    random = choice(list_of_tokens)
                    encoded_text += (random,)
                    break
        decoded_text = self._word_processor.decode(encoded_text)
        if not decoded_text:
            raise ValueError('Decoding is not working.')
        return decoded_text


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
        self.generator_types = {self.greedy: 'Greedy Generator', self.top_p: 'Top-P Generator',
                                self.beam_search: 'Beam Search Generator'}

    def get_conversion_generator_type(self, generator_type: int) -> str:  # type: ignore
        """
        Retrieve string type of generator.

        Args:
            generator_type (int): Numeric type of the generator

        Returns:
            (str): Name of the generator.
        """
        return self.generator_types[generator_type]


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
        return (f'Perplexity score: {self.__perplexity}\n'
                f'{GeneratorTypes().get_conversion_generator_type(self.__type)}\n'
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
        if not isinstance(generated_text, str) or generated_text is None or len(generated_text) == 0:
            raise ValueError('Inappropriate input or input argument is empty')
        encoded_text = self._word_processor.encode(generated_text)
        if encoded_text is None or len(encoded_text) == 0:
            raise ValueError('Could not encode')
        ngram_size = self._language_model.get_n_gram_size()
        sum_log = 0.0
        for i in range(ngram_size - 1, len(encoded_text)):
            context = encoded_text[i - ngram_size + 1: i]
            next_tokens = self._language_model.generate_next_token(context)
            if next_tokens is None or len(next_tokens) == 0:
                raise ValueError('None is returned')
            probability = next_tokens.get(encoded_text[i])
            if probability:
                sum_log += math.log(probability)
        if not sum_log:
            raise ValueError('Sum is 0')
        return math.exp(-sum_log / (len(encoded_text) - ngram_size))

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
        if not isinstance(seq_len, int) or not isinstance(prompt, str):
            raise ValueError('Inappropriate input')
        if prompt is None:
            raise ValueError('input argument is empty')
        if seq_len <= 0:
            raise ValueError('seq_len is a negative int')
        new_list = []
        for num, generator in self._generators.items():
            generated_text = generator.run(seq_len, prompt)
            if generated_text is None:
                continue
            perplexity = self._calculate_perplexity(generated_text)
            if perplexity is None:
                raise ValueError('None is returned')
            new_list.append(GenerationResultDTO(generated_text, perplexity, num))
        new_list.sort(key=lambda x: (x.get_perplexity(), x.get_type()))
        return new_list


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

    def provide_questions(self) -> list[tuple[str, int]]:  # type: ignore
        """
        Provide questions for an exam.

        Returns:
            list[tuple[str, int]]:
                List in the form of [(question, position of the word to be filled)]
        """

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

    def get_generator_type(self) -> str:  # type: ignore
        """
        Retrieve generator type.

        Returns:
            str: Generator type
        """
