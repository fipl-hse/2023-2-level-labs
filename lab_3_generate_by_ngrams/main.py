"""
Lab 3.

Beam-search and natural language generation evaluation
"""
import json
import math
# pylint:disable=too-few-public-methods
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
        if not isinstance(text, str) or not text or not any(char.isalpha() for char in text):
            return None

        splited_text = text.split()
        final_text = []
        for word in splited_text:
            clear_word = [char.lower() for char in word if char.isalpha()]
            if clear_word:
                final_text.extend(clear_word)
                final_text.append(self._end_of_word_token)

        if text[-1].isalnum():
            del final_text[-1]

        return tuple(final_text)

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
        if not isinstance(element_id, int) or element_id not in self._storage.values():
            return None

        tokens = list(filter(lambda x: x[1] == element_id, self._storage.items()))
        return tokens[0][0]

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
        if not (isinstance(text, str) and text):
            return None

        encoded_corpus = []
        tokenized = self._tokenize(text)
        if not tokenized:
            return None
        for token in tokenized:
            self._put(token)
            encoded_corpus_num = self.get_id(token)
            if not isinstance(encoded_corpus_num, int):
                return None
            encoded_corpus.append(encoded_corpus_num)
        return tuple(encoded_corpus)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or len(element) != 1 \
                or element in self._storage:
            return None

        self._storage.update({element: len(self._storage)})
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
        if not isinstance(encoded_corpus, tuple) or encoded_corpus is None:
            return None

        decoded_text = self._decode(encoded_corpus)
        if decoded_text is None:
            return None
        processed_text = self._postprocess_decoded_text(decoded_text)
        if processed_text is None:
            return None

        return processed_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """

        if not(isinstance(content, dict) and content):
            return None

        for key in content['freq']:
            for element in key:
                if element.isalpha():
                    self._put(element.lower())
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
        if not (isinstance(corpus, tuple) and corpus):
            return None

        decoded_corpus = []
        for i, element in enumerate(corpus):
            decoded_element = self.get_token(element)
            if decoded_element is None:
                return None
            decoded_corpus.append(decoded_element)
        if not decoded_corpus:
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
        if not (isinstance(decoded_corpus, tuple) and decoded_corpus):
            return None
        decoded_corpus_list = list(decoded_corpus)
        if decoded_corpus_list[-1] == self._end_of_word_token:
            decoded_corpus_list[-1] = "."
        else:
            decoded_corpus_list.append(".")

        for i, token in enumerate(decoded_corpus_list):
            if token == self._end_of_word_token:
                decoded_corpus_list[i] = " "
        return "".join(decoded_corpus_list).capitalize()


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
        if isinstance(frequencies, dict) and frequencies:
            self._n_gram_frequencies = frequencies

    def build(self) -> int:
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1

        In case of corrupt input arguments or methods used return None,
        1 is returned
        """
        if not (isinstance(self._encoded_corpus, tuple) and self._encoded_corpus):
            return 1

        n_grams = self._extract_n_grams(self._encoded_corpus)

        if n_grams is None:
            return 1

        for n_gram in set(n_grams):
            absolute_frequency = n_grams.count(n_gram)
            beginning_frequency = [context[:-1] for context in n_grams].count(n_gram[:-1])

            self._n_gram_frequencies[n_gram] = absolute_frequency / beginning_frequency

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

        if (not (isinstance(sequence, tuple) and sequence) or
                len(sequence) < self._n_gram_size - 1):
            return None

        context = sequence[- self._n_gram_size + 1:]

        next_letter = {}
        for n_gram, freq in self._n_gram_frequencies.items():
            if context == n_gram[:self._n_gram_size - 1]:
                letter = n_gram[self._n_gram_size - 1]
                if letter not in next_letter:
                    next_letter[letter] = freq

        return next_letter

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
        if not (isinstance(encoded_corpus, tuple) and encoded_corpus):
            return None

        encoded_corpus_list = list(encoded_corpus)
        n_grams = []
        for i in range(len(encoded_corpus_list) - self._n_gram_size + 1):
            n_gram = tuple(encoded_corpus_list[i: i + self._n_gram_size])
            n_grams.append(n_gram)

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
        if not (isinstance(seq_len, int) and isinstance(prompt, str) and prompt):
            return None

        encoded_prompt = self._text_processor.encode(prompt)

        size = self._model.get_n_gram_size()

        if size is None:
            return None
        if encoded_prompt is None:
            return None

        while seq_len > 0:
            possible_letters = self._model.generate_next_token(encoded_prompt[-size + 1:])
            if not possible_letters:
                break
            max_freq = max(possible_letters.values())
            max_freq_letters = [letter for letter, freq in possible_letters.items()
                                if freq == max_freq]
            max_freq_letters = sorted(max_freq_letters)
            encoded_prompt += (max_freq_letters[0],)

            seq_len -= 1

        decoded_prompt = self._text_processor.decode(encoded_prompt)
        if decoded_prompt is None:
            return None

        return decoded_prompt


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

        if not (isinstance(sequence, tuple) and sequence):
            return None

        next_tokens = self._model.generate_next_token(sequence)

        if next_tokens is None:
            return None

        if not next_tokens:
            return []

        return sorted(list(next_tokens.items()),
                             key=lambda x: (-x[1], -x[0]))[:self._beam_width]

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

        if not (isinstance(sequence, tuple) and isinstance(next_tokens, list)
                and isinstance(sequence_candidates, dict) and sequence
                and next_tokens and sequence_candidates and len(next_tokens) <= self._beam_width
                and sequence in sequence_candidates):
            return None

        for token in next_tokens:
            seq = sequence + (token[0],)
            sequence_candidates[seq] = sequence_candidates[sequence] - math.log(token[1])
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

        if not (isinstance(sequence_candidates, dict) and sequence_candidates):
            return None

        sorted_sequences = sorted(sequence_candidates.items(), key=lambda x: (x[1], x[0]))

        return dict(sorted_sequences[:self._beam_width])


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

        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(self._beam_width, language_model)
        self._language_model = language_model

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

        if not (isinstance(prompt, str) and isinstance(seq_len, int) and seq_len and prompt):
            return None

        encoded_sequence = self._text_processor.encode(prompt)

        if encoded_sequence is None:
            return None

        sequence_candidates = {encoded_sequence: 0.0}

        for i in range(seq_len):
            new_seq = dict(sequence_candidates)
            for sequence in sequence_candidates:
                probable_letters = self._get_next_token(sequence)
                if not probable_letters:
                    return None
                probable_seq = self.beam_searcher.continue_sequence(sequence,
                                                                    probable_letters, new_seq)
                if not probable_seq:
                    return self._text_processor.decode(sorted(sequence_candidates,
                                                              key=lambda x: x[1])[0])
                top_seq = self.beam_searcher.prune_sequence_candidates(new_seq)
                if top_seq is None:
                    return None
                sequence_candidates = top_seq
        sorted_top_seq = sorted(sequence_candidates, key=lambda x: x[1])
        decoded_seq = self._text_processor.decode(sorted_top_seq[0])

        return decoded_seq

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

        if not (isinstance(sequence_to_continue, tuple) and sequence_to_continue):
            return None

        next_token = self.beam_searcher.get_next_token(sequence_to_continue)

        if next_token is None:
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
        self._json_path = json_path
        self._eow_token = eow_token
        self._text_processor = TextProcessor(self._eow_token)

        with open(self._json_path, "r", encoding="utf-8") as text_file:
            text = json.load(text_file)
            self._content = text
            self._text_processor.fill_from_ngrams(self._content)

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
        if not(isinstance(n_gram_size, int) and n_gram_size and 2 <= n_gram_size <= 5):
            return None

        ngrams = {}
        for ngram in self._content['freq']:
            encoded = []
            for token in ngram:
                if token.isspace():
                    encoded.append(0)
                elif token.isalpha():
                    token_id = self._text_processor.get_id(token.lower())
                    if not token_id:
                        continue
                    encoded.append(token_id)

            if tuple(encoded) not in ngrams:
                ngrams[tuple(encoded)] = 0.0

            ngrams[tuple(encoded)] += self._content['freq'][ngram]

        clean_ngrams = {}
        for ngram, freq in ngrams.items():
            if isinstance(ngram, tuple) and len(ngram) == n_gram_size:
                same_context = [context_freq for context, context_freq in ngrams.items()
                                if context[-n_gram_size:-1] == ngram[-n_gram_size:-1]]
                clean_ngrams[ngram] = freq / sum(same_context)

        ngram_model = NGramLanguageModel(None, n_gram_size)
        ngram_model.set_n_grams(clean_ngrams)

        return ngram_model

    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """
        return self._text_processor


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
        self._language_models = {model.get_n_gram_size(): model for model in language_models}
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
        if not (isinstance(seq_len, int) and isinstance(prompt, str) and prompt
        ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None

        for i in range(seq_len):
            candidates = self._get_next_token(encoded_prompt)
            if not candidates:
                break

            best_candidate = [token for token, freq in candidates.items()
                              if freq == max(candidates.values())]
            encoded_prompt += (best_candidate[0],)
        decoded_sequence = self._text_processor.decode(encoded_prompt)

        if not decoded_sequence:
            return None

        return decoded_sequence

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

        if not (isinstance(sequence_to_continue, tuple) and sequence_to_continue
                and self._language_models
        ):
            return None

        ngram_sizes = sorted(self._language_models.keys(), reverse=True)

        for ngram_size in ngram_sizes:
            ngram_model = self._language_models[ngram_size]

            candidates = ngram_model.generate_next_token(sequence_to_continue)

            if candidates is None:
                return None

            if len(candidates) > 0:
                probabilities = {token: freq / sum(candidates.values())
                                 for token, freq in candidates.items()}
                return probabilities

        return None
