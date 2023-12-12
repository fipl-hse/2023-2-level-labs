"""
Lab 4.

Top-p sampling generation and filling gaps with ngrams
"""
# pylint:disable=too-few-public-methods, too-many-arguments
import json
import math
import random

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
            raise ValueError("Incorrect input")

        for symbol in ['.', '!', '?']:
            text = text.replace(symbol, f" {self._end_of_word_token} ")

        clean_text = []
        for word in text.lower().split():
            if word == self._end_of_word_token or word.isalpha() or word.isspace():
                clean_text.append(word)
            else:
                clean_word = []
                for alpha in list(word):
                    if alpha.isalpha():
                        clean_word.append(alpha)
                if clean_word:
                    clean_text.append("".join(clean_word))

        return tuple(clean_text)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): Element to put into storage

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(element, str) or not element:
            raise ValueError("Incorrect input")

        if element not in self._storage:
            self._storage.update({element: len(self._storage)})

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
            raise ValueError("Incorrect input")

        postprocessed = " ".join(decoded_corpus).replace(f" {self._end_of_word_token}", ".")
        postprocessed = ". ".join([sentence.capitalize()
                                   for sentence in postprocessed.split(". ")])

        if postprocessed[-1] != ".":
            postprocessed += "."

        return postprocessed


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
        if not isinstance(seq_len, int) or seq_len < 0 or not isinstance(prompt, str) or not prompt:
            raise ValueError("Incorrect input")

        encoded = self._word_processor.encode(prompt)
        if not encoded:
            raise ValueError("Encode returned nothing")

        cur_seq_len = 0
        while cur_seq_len < seq_len:
            next_tokens = self._model.generate_next_token(encoded)
            if next_tokens is None:
                raise ValueError("Generate next token returned nothing")
            if not next_tokens:
                break

            sorted_next_tokens = sorted(next_tokens.items(),
                                        key=lambda item: (item[1], item[0]), reverse=True)

            sum_p = 0
            taken_n = []
            for token in sorted_next_tokens:
                if sum_p < self._p_value:
                    sum_p += token[1]
                    taken_n.append(token[0])
            chosen = random.choice(taken_n)
            encoded += (chosen,)
            cur_seq_len += 1

        decoded = self._word_processor.decode(encoded)
        if decoded is None:
            raise ValueError("Decode returned nothing")

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
        generators = ['Greedy Generator', 'Top-P Generator', 'Beam Search Generator']
        return generators[generator_type]


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
        return f"Perplexity score: {self.__perplexity}\n" \
               f"{GeneratorTypes().get_conversion_generator_type(self.__type)}\n" \
               f"Text: {self.__text}\n"


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
        if not isinstance(generated_text, str) or not generated_text:
            raise ValueError("Incorrect input")

        encoded = self._word_processor.encode(generated_text)
        if not encoded:
            raise ValueError("Encode returned nothing")

        ngram_size = self._language_model.get_n_gram_size()
        sum_logs = 0.0
        for i in range(ngram_size - 1, len(encoded)):
            context = tuple(encoded[i - ngram_size + 1: i])
            next_tokens = self._language_model.generate_next_token(context)
            if not next_tokens:
                raise ValueError("Generate next token returned nothing")

            prob = next_tokens.get(encoded[i])
            if prob:
                sum_logs += math.log(prob)
        if not sum_logs:
            raise ValueError("No logs found")
        return math.exp(-sum_logs / (len(encoded) - ngram_size))

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
        if not isinstance(seq_len, int) or seq_len < 0 or not isinstance(prompt, str) or not prompt:
            raise ValueError("Incorrect input")
        calculations = []
        for num_type, name in self._generators.items():
            generated = name.run(prompt=prompt, seq_len=seq_len)
            if not generated:
                raise ValueError("Nothing was generated")

            perplexity = self._calculate_perplexity(generated)
            if not perplexity:
                raise ValueError("No calculated perplexity")

            calculations.append(GenerationResultDTO(generated, perplexity, num_type))
        return sorted(calculations, key=lambda item: (perplexity, num_type))


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
        self._json_path = json_path
        self._questions_and_answers = self._load_from_json()

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
        if not isinstance(self._json_path, str) or not self._json_path \
                or self._json_path[-5:] != ".json":
            raise ValueError("Incorrect input")

        questions_and_answers = {}
        with open(self._json_path, 'r', encoding='utf-8') as f:
            test_dict = json.load(f)
        if not isinstance(test_dict, list):
            raise ValueError("Incorrect type")
        for task in test_dict:
            questions_and_answers.update({(task["question"], task["location"]): task["answer"]})
        return questions_and_answers

    def provide_questions(self) -> list[tuple[str, int]]:  # type: ignore
        """
        Provide questions for an exam.

        Returns:
            list[tuple[str, int]]:
                List in the form of [(question, position of the word to be filled)]
        """
        questions = list(self._questions_and_answers.keys())
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
        if not isinstance(answers, dict) or not answers:
            raise ValueError("Incorrect input")
        correct_answers = list(self._questions_and_answers.values())
        students_answers = list(answers.values())
        correct = 0
        for i, answer in enumerate(correct_answers):
            if answer == students_answers[i]:
                correct += 1
        return correct / len(correct_answers)


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
        generators = (GreedyTextGenerator(language_model, word_processor),
                      TopPGenerator(language_model, word_processor, 0.5),
                      BeamSearchTextGenerator(language_model, word_processor, 5))
        self._generator = generators[self._generator_type]

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
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Incorrect input")

        final_dict = {}
        for (question, position) in tasks:
            left_context = question[:position]
            right_context = question[position:]
            next_seq = self._generator.run(seq_len=1, prompt=left_context)
            if not next_seq:
                raise ValueError("Nothing was generated")
            if next_seq[-1] == ".":
                next_seq = next_seq[:-1] + " "
            answer = next_seq + right_context
            final_dict.update({question: answer})
        return final_dict

    def get_generator_type(self) -> str:  # type: ignore
        """
        Retrieve generator type.

        Returns:
            str: Generator type
        """
        generator = GeneratorTypes()
        return generator.get_conversion_generator_type(self._generator_type)
