"""
Lab 3.

Beam-search and natural language generation evaluation
"""
# pylint:disable=too-few-public-methods
import math
from typing import Optional


class TextProcessor:
    """
    Handle text tokenization, encoding and decoding.

    Attributes:
        _end_of_word_token (str): A token denoting word boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def __init__(self, end_of_word_token: str) -> None:
        """
        Initialize an instance of LetterStorage.

        Args:
            end_of_word_token (str): A token denoting word boundary
        """
        self._end_of_word_token = end_of_word_token
        self._storage = {end_of_word_token: 0}

    def _tokenize(self, text: str) -> Optional[tuple[str, ...]]:
        """
        Tokenize text into unigrams, separating words with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...]: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str) or not text:
            return None
        text_list = []
        for symbol in text.lower():
            if symbol.isspace() and text_list[-1] != self._end_of_word_token:
                text_list.append(self._end_of_word_token)
            elif symbol.isalpha():
                text_list.append(symbol)
        if not text[-1].isalnum():
            text_list.append(self._end_of_word_token)
        if text_list.count(self._end_of_word_token) == len(text_list):
            return None
        return tuple(text_list)

    def get_id(self, element: str) -> Optional[int]:
        """
        Retrieve a unique identifier of an element.

        Args:
            element (str): String element to retrieve identifier for

        Returns:
            int: Integer identifier that corresponds to the given element

        In case of corrupt input arguments or arguments not included in storage,
        None is returned
        """
        if not isinstance(element, str) or element not in self._storage:
            return None
        return self._storage[element]

    def get_end_of_word_token(self) -> str:
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return self._end_of_word_token

    def get_token(self, element_id: int) -> Optional[str]:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """
        if not isinstance(element_id, int):
            return None
        storage_inverted = {symbol: ind for ind, symbol in self._storage.items()}
        if element_id not in storage_inverted:
            return None
        return storage_inverted[element_id]

    def encode(self, text: str) -> Optional[tuple[int, ...]]:
        """
        Encode text.

        Tokenize text, assign each symbol an integer identifier and
        replace letters with their ids.

        Args:
            text (str): An original text to be encoded

        Returns:
            tuple[int, ...]: Processed text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None
        tokenized_text = self._tokenize(text)
        if not tokenized_text:
            return None
        encoded_corpus = []
        for symbol in tokenized_text:
            self._put(symbol)
            ind = self.get_id(symbol)
            if not isinstance(ind, int):
                return None
            encoded_corpus.append(ind)
        if None in encoded_corpus:
            return None
        return tuple(encoded_corpus)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or len(element) != 1:
            return None
        if element not in self._storage:
            self._storage[element] = len(self._storage)
        return None

    def decode(self, encoded_corpus: tuple[int, ...]) -> Optional[str]:
        """
        Decode and postprocess encoded corpus by converting integer identifiers to string.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(encoded_corpus, tuple) or not encoded_corpus:
            return None
        decoded_corpus = self._decode(encoded_corpus)
        if not decoded_corpus:
            return None
        decoded_text = self._postprocess_decoded_text(decoded_corpus)
        if not decoded_text:
            return None
        return decoded_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
            return None
        for n_gram in content['freq']:
            for letter in n_gram:
                if letter.isalpha():
                    self._put(letter)
        return None

    def _decode(self, corpus: tuple[int, ...]) -> Optional[tuple[str, ...]]:
        """
        Decode sentence by replacing ids with corresponding letters.

        Args:
            corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[str, ...]: Sequence with decoded tokens

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(corpus, tuple) or not corpus:
            return None
        decoded_corpus = []
        for num in corpus:
            letter = self.get_token(num)
            if not isinstance(letter, str):
                return None
            decoded_corpus.append(letter)
        if None in decoded_corpus:
            return None
        return tuple(decoded_corpus)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> Optional[str]:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            return None
        decoded_list = list(decoded_corpus)
        if decoded_list[-1] == self._end_of_word_token:
            decoded_list = decoded_list[:-1]
        decoded_str = ''.join(decoded_list).replace(self._end_of_word_token, ' ') + '.'
        return decoded_str.capitalize()

class NGramLanguageModel:
    """
    Store language model by n_grams, predict the next token.

    Attributes:
        _n_gram_size (int): A size of n-grams to use for language modelling
        _n_gram_frequencies (dict): Frequencies for n-grams
        _encoded_corpus (tuple): Encoded text
    """

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an instance of NGramLanguageModel.

        Args:
            encoded_corpus (tuple): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        self._encoded_corpus = encoded_corpus
        self._n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

    def get_n_gram_size(self) -> int:
        """
        Retrieve value stored in self._n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """
        return self._n_gram_size

    def set_n_grams(self, frequencies: dict) -> None:
        """
        Setter method for n-gram frequencies.

        Args:
            frequencies (dict): Computed in advance frequencies for n-grams
        """
        if not isinstance(frequencies, dict) or not frequencies:
            return None
        self._n_gram_frequencies = frequencies
        return None

    def build(self) -> int:
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1

        In case of corrupt input arguments or methods used return None,
        1 is returned
        """
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1
        n_grams = self._extract_n_grams(self._encoded_corpus)
        if not n_grams:
            return 1
        for n_gram in set(n_grams):
            b_freq = len([n_gram_2 for n_gram_2 in n_grams if n_gram_2[:-1] == n_gram[:-1]])
            self._n_gram_frequencies[n_gram] = n_grams.count(n_gram) / b_freq
        return 0

    def generate_next_token(self, sequence: tuple[int, ...]) -> Optional[dict]:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            Optional[dict]: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if (not isinstance(sequence, tuple) or not sequence
            or len(sequence) < self._n_gram_size - 1):
            return None
        context = sequence[-self._n_gram_size + 1:]
        predict = {}
        for n_gram, freq in self._n_gram_frequencies.items():
            if n_gram[:len(context)] == context:
                predict[n_gram[len(context)]] = freq
        return predict

    def _extract_n_grams(
        self, encoded_corpus: tuple[int, ...]
    ) -> Optional[tuple[tuple[int, ...], ...]]:
        """
        Split encoded sequence into n-grams.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[tuple[int, ...], ...]: A tuple of extracted n-grams

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(encoded_corpus, tuple) or not encoded_corpus:
            return None
        n_grams = []
        for ind, num in enumerate(encoded_corpus[:-1]):
            n_grams.append(tuple(encoded_corpus[ind: ind + self._n_gram_size]))
        return tuple(n_grams)


class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, language_model: NGramLanguageModel, text_processor: TextProcessor) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._model = language_model
        self._text_processor = text_processor

    def run(self, seq_len: int, prompt: str) -> Optional[str]:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (not isinstance(seq_len, int) or
            not isinstance(prompt, str) or not prompt):
            return None
        prompt_encoded = self._text_processor.encode(prompt)
        n_gram_size = self._model.get_n_gram_size()
        if not prompt_encoded or not n_gram_size:
            return None
        list_encoded = list(prompt_encoded)
        for prediction in range(seq_len):
            seq = tuple(list_encoded[-n_gram_size + 1:])
            variants = self._model.generate_next_token(seq)
            if not variants:
                break
            best_choice = sorted([pred for pred, freq in variants.items() if
                                  freq == max(variants.values())])[0]
            list_encoded.append(best_choice)
        predicted = self._text_processor.decode(tuple(list_encoded))
        if not predicted:
            return None
        return predicted


class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model to use for next token prediction
    """

    def __init__(self, beam_width: int, language_model: NGramLanguageModel) -> None:
        """
        Initialize an instance of BeamSearchAlgorithm.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(self, sequence: tuple[int, ...]) -> Optional[list[tuple[int, float]]]:
        """
        Retrieves candidate tokens for sequence continuation.

        The valid candidate tokens are those that are included in the N-gram with.
        Number of tokens retrieved must not be bigger that beam width parameter.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            Optional[list[tuple[int, float]]]: Tokens to use for
            base sequence continuation
            The return value has the following format:
            [(token, probability), ...]
            The return value length matches the Beam Size parameter.

        In case of corrupt input arguments or methods used return None.
        """
        if not isinstance(sequence, tuple) or not sequence:
            return None
        variants = self._model.generate_next_token(sequence)
        if variants is None:
            return None
        if not variants:
            return []
        most_likely = sorted([(token, float(freq)) for token, freq in variants.items()],
                        key=lambda x: x[1], reverse=True)[:self._beam_width]
        return most_likely

    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> Optional[dict[tuple[int, ...], float]]:
        """
        Generate new sequences from the base sequence with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for sequence continuation
            sequence_candidates (dict[tuple[int, ...], dict]): Storage with all sequences generated

        Returns:
            Optional[dict[tuple[int, ...], float]]: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if (not isinstance(sequence, tuple) or not isinstance(next_tokens, list) or
            not isinstance(sequence_candidates, dict) or not sequence):
            return None
        if (not next_tokens or not sequence_candidates or
           sequence not in sequence_candidates or
           len(next_tokens) > self._beam_width):
            return None
        for token in next_tokens:
            sequence_candidates[sequence + tuple([token[0]])] = \
                sequence_candidates[sequence] - math.log(token[1])
        sequence_candidates.pop(sequence)
        return sequence_candidates

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> Optional[dict[tuple[int, ...], float]]:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (int): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float]: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_candidates, dict) or not sequence_candidates:
            return None
        sorted_sequences = []
        candidates_copy = sequence_candidates.copy()
        for candidate in range(len(sequence_candidates)):
            if not candidates_copy:
                break
            most_probable_candidates = [seq for seq, freq in candidates_copy.items() if
                                        freq == min(candidates_copy.values())]
            sorted_sequences.extend(sorted(most_probable_candidates))
            for cand in most_probable_candidates:
                candidates_copy.pop(cand)
        sequences_to_remove = sorted_sequences[self._beam_width:]
        for sequence in sequences_to_remove:
            sequence_candidates.pop(sequence)
        return sequence_candidates


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (tuple[NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
        _beam_width (NGramLanguageModel): Beam width parameter for generation
        beam_searcher (NGramLanguageModel): Searcher instances for each language model
    """

    def __init__(
        self,
        language_model: NGramLanguageModel,
        text_processor: TextProcessor,
        beam_width: int,
    ):
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
            beam_width (int): Beam width parameter for generation
        """
        self._language_model = language_model
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(self._beam_width, self._language_model)

    def run(self, prompt: str, seq_len: int) -> Optional[str]:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (not isinstance(prompt, str) or not prompt or
            not isinstance(seq_len, int) or seq_len <= 0):
            return None
        prompt_encoded = self._text_processor.encode(prompt)
        if not prompt_encoded:
            return None
        sequence_candidates = {prompt_encoded: 0.0}
        for prediction in range(seq_len):
            candidates_copy = sequence_candidates.copy()
            for sequence in sequence_candidates:
                tokens = self._get_next_token(sequence)
                if not tokens:
                    return None
                n_sequence_candidates = self.beam_searcher.continue_sequence(sequence,
                                                                             tokens,
                                                                             candidates_copy)
                if not n_sequence_candidates:
                    for_decode = sorted([seq for seq, prob in candidates_copy.items() if
                                         prob == min(candidates_copy.values())])[0]
                    prompt_decoded = self._text_processor.decode(tuple(for_decode))
                    return prompt_decoded
            best_choice = self.beam_searcher.prune_sequence_candidates(candidates_copy)
            if not best_choice:
                return None
            sequence_candidates = best_choice
        for_decode = sorted([seq for seq, prob in sequence_candidates.items() if
                             prob == min(sequence_candidates.values())])[0]
        prompt_decoded = self._text_processor.decode(tuple(for_decode))
        return prompt_decoded

    def _get_next_token(
        self, sequence_to_continue: tuple[int, ...]
    ) -> Optional[list[tuple[int, float]]]:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            Optional[list[tuple[int, float]]]: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None
        next_token = self.beam_searcher.get_next_token(sequence_to_continue)
        if not next_token:
            return None
        return next_token


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, json_path: str, eow_token: str) -> None:
        """
        Initialize reader instance.

        Args:
            json_path (str): Local path to assets file
            eow_token (str): Special token for text processor
        """

    def load(self, n_gram_size: int) -> Optional[NGramLanguageModel]:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned from digits and punctuation,
        their length must match n_gram_size, and spaces must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel: Built language model.

        In case of corrupt input arguments or unexpected behaviour of methods used, return 1.
        """

    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """


class BackOffGenerator:
    """
    Language model for back-off based text generation.

    Attributes:
        _language_models (dict[int, NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
    """

    def __init__(
        self,
        language_models: tuple[NGramLanguageModel, ...],
        text_processor: TextProcessor,
    ):
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel]): Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """

    def run(self, seq_len: int, prompt: str) -> Optional[str]:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> Optional[dict[int, float]]:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            Optional[dict[int, float]]: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
