"""
Lab 3.

Beam-search and natural language generation evaluation
"""
# pylint:disable=too-few-public-methods
from typing import Optional
import math
import operator
import json


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
        self._storage = {self._end_of_word_token: 0}

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
        if not isinstance(text, str):
            return None

        tokenized_text = []
        for index, element in enumerate(text.lower()):
            if element.isalpha():
                tokenized_text.append(element)
            elif element.isdigit():
                pass
            elif not index:
                pass
            elif tokenized_text[-1] != self._end_of_word_token:
                tokenized_text.append(self._end_of_word_token)

        if not tokenized_text:
            return None

        return tuple(tokenized_text)

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

        inv_storage = {identifier: element for element, identifier in self._storage.items()}

        if element_id not in inv_storage:
            return None

        return inv_storage[element_id]

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
        if not isinstance(text, str) or not text:
            return None

        tokenized_text = self._tokenize(text)
        if tokenized_text is None:
            return None

        for token in tokenized_text:
            self._put(token)

        processed_text = []
        for token in tokenized_text:
            if self.get_id(token) is None:
                return None
            processed_text.append(self.get_id(token))

        for ident in processed_text:
            if not isinstance(ident, int):
                return None

        return tuple(processed_text)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if isinstance(element, str) and len(element) == 1 and element not in self._storage:
            self._storage[element] = len(self._storage)

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
        if decoded_corpus is None:
            return None

        postprocessed_text = self._postprocess_decoded_text(decoded_corpus)
        if postprocessed_text is None:
            return None

        return postprocessed_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
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
        if not isinstance(corpus, tuple) or not corpus:
            return None

        decoded_corpus = []
        for identifier in corpus:
            if self.get_token(identifier) is not None:
                decoded_corpus.append(self.get_token(identifier))

        if decoded_corpus is None or not decoded_corpus:
            return None

        for token in decoded_corpus:
            if not isinstance(token, str):
                return None

        return (*decoded_corpus, )

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

        postprocessed_text = ''
        for index, token in enumerate(decoded_corpus):
            if not index:
                postprocessed_text += token.upper()
            elif token == self._end_of_word_token:
                if index == len(decoded_corpus) - 1:
                    postprocessed_text += '.'
                else:
                    postprocessed_text += ' '
            else:
                postprocessed_text += token
        if postprocessed_text[-1] != '.':
            postprocessed_text += '.'

        return postprocessed_text


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

        self._n_gram_frequencies.update(frequencies)
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
        if n_grams is None:
            return 1

        for n_gram in n_grams:
            abs_freq = n_grams.count(n_gram)
            rel_freq = 0
            for n_gram_to_compare in n_grams:
                if n_gram_to_compare[:self._n_gram_size - 1] == n_gram[:self._n_gram_size - 1]:
                    rel_freq += 1
            self._n_gram_frequencies[n_gram] = abs_freq / rel_freq

        if not self._n_gram_frequencies:
            return 1

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
        if not isinstance(sequence, tuple) or not sequence or len(sequence) < self._n_gram_size - 1:
            return None

        possible_tokens = {}

        context = sequence[-(self._n_gram_size - 1):]

        sort_data = dict(sorted(self._n_gram_frequencies.items(), key=lambda x: (x[1], list(x[0]))))

        for n_gram, freq in sort_data.items():
            if n_gram[:self._n_gram_size - 1] == context:
                possible_tokens[n_gram[-1]] = freq

        return possible_tokens

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
        return tuple(encoded_corpus[index:index + self._n_gram_size]
                     for index in range(len(encoded_corpus) - 1))


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
        if not isinstance(seq_len, int) or not isinstance(prompt, str) or not prompt:
            return None

        n_gram_size = self._model.get_n_gram_size()
        encoded = self._text_processor.encode(prompt)
        if not encoded or not n_gram_size:
            return None

        max_freq = []
        for _ in range(seq_len):
            tokens = self._model.generate_next_token(encoded[-n_gram_size + 1:])
            if not tokens:
                break
            max_freq.append(max(tokens.values()))
            candidates_max = filter(lambda token_freq: token_freq[1] == max_freq[-1],
                                    tokens.items())
            encoded += (sorted(candidates_max)[0][0],)

        text = self._text_processor.decode(encoded)
        if not text:
            return None
        return text


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

        possible_tokens = self._model.generate_next_token(sequence)
        if possible_tokens is None:
            return None
        if not possible_tokens:
            return []

        possible_tokens_list = list(possible_tokens.items())
        sorted_tokens_list = sorted(possible_tokens_list,
                                    key=operator.itemgetter(1, 0), reverse=True)
        best_tokens = sorted_tokens_list[:self._beam_width]

        return best_tokens

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
        if not isinstance(sequence, tuple) or not isinstance(next_tokens, list) \
                or not isinstance(sequence_candidates, dict):
            return None
        if not sequence or not next_tokens or not sequence_candidates:
            return None
        if sequence not in sequence_candidates:
            return None
        if len(next_tokens) > self._beam_width:
            return None

        copy_seq_candidates = sequence_candidates.copy()
        list_sequence = list(sequence)

        for token in next_tokens:
            list_sequence.append(token[0])
            possible_seq = tuple(list_sequence)
            freq = sequence_candidates[sequence] - math.log(token[-1])
            copy_seq_candidates[possible_seq] = freq
            list_sequence = list_sequence[:-1]

        copy_seq_candidates.pop(sequence)

        return copy_seq_candidates

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

        sorted_sequences = sorted(sequence_candidates.items(), key=operator.itemgetter(1, 0))
        result = {}
        for sequence in sorted_sequences[:self._beam_width]:
            result[sequence[0]] = sequence[1]

        return result


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
        self._language_model = language_model
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
        if (not isinstance(seq_len, int) or seq_len <= 0 or
                not isinstance(prompt, str) or not prompt):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        candidates = {encoded_prompt: 0.0}

        updated_candidates = candidates.copy()

        for _ in range(seq_len):
            candidates = updated_candidates
            not_sorted_candidates = {}

            for sequence in candidates:
                next_tokens = self._get_next_token(sequence)
                if not next_tokens:
                    return None

                possible_sequences = self.beam_searcher.continue_sequence(sequence,
                                                                          next_tokens,
                                                                          updated_candidates)
                if not possible_sequences:
                    return self._text_processor.decode(sorted(tuple(updated_candidates),
                                                              key=lambda x: x[1])[0])

                not_sorted_candidates.update(possible_sequences)

            for candidate in candidates:
                if candidate in not_sorted_candidates:
                    del not_sorted_candidates[candidate]

            sorted_candidates = self.beam_searcher.prune_sequence_candidates(not_sorted_candidates)
            if not sorted_candidates:
                return None

            updated_candidates = sorted_candidates

        return self._text_processor.decode(sorted(tuple(updated_candidates), key=lambda x: x[1])[0])

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

        return self.beam_searcher.get_next_token(sequence_to_continue)


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

        with open(self._json_path, 'r', encoding='utf-8') as file:
            json_text = json.load(file)
        self._content = json_text
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
        if not isinstance(n_gram_size, int) or not n_gram_size or n_gram_size < 2:
            return None

        frequencies = {}
        for key in self._content['freq']:
            encoded = []
            for element in key:
                if element == ' ':
                    encoded.append(0)
                elif element.isalpha():
                    ident = self._text_processor.get_id(element.lower())
                    if isinstance(ident, int):
                        encoded.append(ident)

            if tuple(encoded) not in frequencies:
                frequencies[tuple(encoded)] = 0
            frequencies[tuple(encoded)] += self._content['freq'][key]

        right_ngrams = {}
        for key, value in frequencies.items():
            if len(key) == n_gram_size:
                abs_freq = value
                rel_freq = 0
                for ngram_to_compare, frequency in frequencies.items():
                    if ngram_to_compare[:n_gram_size - 1] == key[:n_gram_size - 1]:
                        rel_freq += frequency
                freq = abs_freq / rel_freq
                right_ngrams[key] = freq

        lang_model = NGramLanguageModel(None, n_gram_size)
        lang_model.set_n_grams(right_ngrams)
        return lang_model

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
        self._language_models = {lang_model.get_n_gram_size(): lang_model
                                 for lang_model in language_models}
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
        if not isinstance(seq_len, int) or not isinstance(prompt, str) or not prompt:
            return None

        encoded_sequence = self._text_processor.encode(prompt)
        if not encoded_sequence:
            return None
        for _ in range(seq_len):
            candidates = self._get_next_token(encoded_sequence)
            if not candidates:
                break
            max_probability = max(candidates.values())
            best_candidates = []
            for element, freq in candidates.items():
                if freq == max_probability:
                    best_candidates.append(element)
            encoded_sequence += (best_candidates[0],)
        decoded = self._text_processor.decode(encoded_sequence)
        if not decoded:
            return None
        return decoded

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
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None

        ngram_size_list = list(self._language_models.keys())
        if not ngram_size_list:
            return None
        ngram_size_list.sort(reverse=True)
        for ngram_size in ngram_size_list:
            candidates = self._language_models[ngram_size].generate_next_token(sequence_to_continue)
            if candidates is None:
                return None
            if not candidates:
                continue
            return candidates

        return None
