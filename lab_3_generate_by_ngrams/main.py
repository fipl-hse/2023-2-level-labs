"""
Lab 3.

Beam-search and natural language generation evaluation
"""
# pylint:disable=too-few-public-methods
import json
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
        if not isinstance(text, str) or len(text) == 0:
            return None

        text_words = text.lower().split()
        tokens = []
        for word in text_words:
            len_prev = len(tokens)
            for alpha in word:
                if alpha.isalpha():
                    tokens.append(alpha)
            if len(tokens) > len_prev:
                tokens.append(self._end_of_word_token)

        if not tokens:
            return None

        if text[-1].isdigit() or text[-1].isalpha():
            tokens = tokens[:-1]

        return tuple(tokens)

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
        if not isinstance(element, str):
            return None

        if element not in self._storage:
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

        for token, ident in self._storage.items():
            if ident == element_id:
                return token

        return None

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

        tokenized_text = self._tokenize(text)
        if not tokenized_text:
            return None

        encoded_corpus = []
        for token in tokenized_text:
            self._put(token)
            token_id = self.get_id(token)
            if not isinstance(token_id, int):
                return None
            encoded_corpus.append(token_id)

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
        if not isinstance(encoded_corpus, tuple):
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
        if not isinstance(content, dict) or len(content) == 0:
            return None

        for n_gram in content['freq']:
            for token in n_gram.lower():
                if token.isalpha():
                    self._put(token)

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
        if not isinstance(corpus, tuple) or len(corpus) == 0:
            return None

        decoded_corpus = []
        for ident in corpus:
            token = self.get_token(ident)
            if not token:
                return None
            decoded_corpus.append(token)

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
        if not isinstance(decoded_corpus, tuple) or len(decoded_corpus) == 0:
            return None

        decoded_text = ''.join(decoded_corpus)
        decoded_text = decoded_text.replace('_', ' ')
        decoded_text = decoded_text.capitalize()
        if decoded_text[-1] == ' ':
            decoded_text = decoded_text[:-1]

        decoded_text += '.'

        return decoded_text


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
        if not isinstance(frequencies, dict) or len(frequencies) == 0:
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
        if not isinstance(self._encoded_corpus, tuple) or len(self._encoded_corpus) == 0:
            return 1

        n_grams = self._extract_n_grams(self._encoded_corpus)
        if not n_grams or not isinstance(n_grams, tuple):
            return 1

        context_freq_dict = {}

        for n_gram in set(n_grams):
            absolute_freq = n_grams.count(n_gram)
            context = n_gram[:-1]
            if context not in context_freq_dict:
                context_freq_dict[context] = sum(1 for n_gram_1 in n_grams
                                                 if n_gram_1[:-1] == context)
            context_freq = context_freq_dict[context]

            self._n_gram_frequencies[n_gram] = absolute_freq / context_freq

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
        if (not isinstance(sequence, tuple) or len(sequence) == 0
                or len(sequence) < self._n_gram_size - 1):
            return None

        token_frequencies = {}

        context_size = self._n_gram_size - 1
        context = sequence[-context_size:]

        for n_gram, freq in self._n_gram_frequencies.items():
            if n_gram[:-1] == context:
                token = n_gram[-1]
                if token not in token_frequencies:
                    token_frequencies[token] = freq

        return token_frequencies

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
        if not isinstance(encoded_corpus, tuple) or len(encoded_corpus) == 0:
            return None

        n_gram_size = self._n_gram_size
        n_grams = []
        for i in range(len(encoded_corpus) - n_gram_size + 1):
            n_gram = tuple(encoded_corpus[i:i + n_gram_size])
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
        if not (isinstance(seq_len, int) and isinstance(prompt, str)) or len(prompt) == 0:
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        n_gram_size = self._model.get_n_gram_size()
        if not (encoded_prompt and n_gram_size):
            return None

        while seq_len > 0:
            tokens = self._model.generate_next_token(encoded_prompt[-n_gram_size + 1:])
            if not tokens:
                break

            max_freq = max(tokens.values())
            max_freq_tokens = [token for token, freq in tokens.items() if freq == max_freq]
            max_freq_tokens = sorted(max_freq_tokens)
            encoded_prompt += (max_freq_tokens[0],)

            seq_len -= 1

        decoded_text = self._text_processor.decode(encoded_prompt)
        if not decoded_text:
            return None

        return decoded_text


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

        tokens = self._model.generate_next_token(sequence)
        if tokens is None:
            return None
        if not tokens:
            return []

        token_pairs = sorted(list(tokens.items()), key=lambda x: x[1], reverse=True)
        if len(token_pairs) <= self._beam_width:
            return token_pairs

        return token_pairs[:self._beam_width]

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
            new_sequence = sequence + (token[0],)
            new_freq = sequence_candidates[sequence] - math.log(token[1])
            sequence_candidates[new_sequence] = new_freq

        del sequence_candidates[sequence]

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

        return dict(sorted(list(sequence_candidates.items()),
                           key=lambda x: x[1])[:self._beam_width])


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
        if not (isinstance(prompt, str) and isinstance(seq_len, int) and prompt and seq_len):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None

        sequence_candidates = {encoded_prompt: 0.0}

        for i in range(seq_len):
            new_sequence_candidates = sequence_candidates.copy()

            for sequence in sequence_candidates:
                tokens = self._get_next_token(sequence)
                if not tokens:
                    return None

                continuation_candidates = self.beam_searcher.continue_sequence(
                    sequence, tokens, new_sequence_candidates)
                if not continuation_candidates:
                    return self._text_processor.decode(sorted(tuple(sequence_candidates),
                                                              key=lambda x: x[1])[0])

            if not new_sequence_candidates:
                break

            best_sequence_candidates = self.beam_searcher.prune_sequence_candidates(
                new_sequence_candidates)

            if not best_sequence_candidates:
                return None
            sequence_candidates = best_sequence_candidates

        decoded = self._text_processor.decode(min(sequence_candidates,
                                                  key=lambda x: sequence_candidates[x]))
        return decoded

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
        if not isinstance(sequence_to_continue, tuple) or len(sequence_to_continue) == 0:
            return None

        tokens = self.beam_searcher.get_next_token(sequence_to_continue)
        if not tokens:
            return None

        return tokens


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
        with open(json_path, 'r', encoding="utf-8") as file:
            self._content = json.load(file)
        self._text_processor = TextProcessor(self._eow_token)
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
        n_grams = {}
        for n_gram in self._content['freq']:
            encoded = []
            for token in n_gram:
                if token.isspace():
                    encoded.append(0)
                elif token.isalpha():
                    token_id = self._text_processor.get_id(token.lower())
                    if not token_id:
                        continue
                    encoded.append(token_id)

            if tuple(encoded) not in n_grams:
                n_grams[tuple(encoded)] = 0.0
            n_grams[tuple(encoded)] += self._content['freq'][n_gram]

        n_grams_cleared_by_size = {}
        for n_gram, freq in n_grams.items():
            if len(n_gram) == n_gram_size and isinstance(n_gram, tuple):
                same_context = [context_freq for context, context_freq in n_grams.items()
                                if context[-n_gram_size:-1] == n_gram[-n_gram_size:-1]]
                n_grams_cleared_by_size[n_gram] = freq / sum(same_context)

        n_gram_model = NGramLanguageModel(None, n_gram_size)
        n_gram_model.set_n_grams(n_grams_cleared_by_size)

        return n_gram_model

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
        if not (
                isinstance(seq_len, int) and isinstance(prompt, str) and prompt
        ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)

        if encoded_prompt is None:
            return None

        iteration = 1
        generated_sequence = list(encoded_prompt)
        while iteration <= seq_len:
            next_token_candidates = None
            for n_gram_size in sorted(self._language_models.keys(), reverse=True):
                next_token_candidates = self._get_next_token(
                    tuple(generated_sequence[-(n_gram_size - 1):]))
                if next_token_candidates is not None and len(next_token_candidates) > 0:
                    break

            if next_token_candidates is None or len(next_token_candidates) == 0:
                break

            max_probability_token = max(next_token_candidates, key=lambda x: next_token_candidates.get(x))
            generated_sequence.append(max_probability_token)

            iteration += 1

        decoded_sequence = self._text_processor.decode(tuple(generated_sequence))

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
                and self._language_models):
            return None

        n_gram_sizes = sorted(self._language_models.keys(), reverse=True)

        for n_gram_size in n_gram_sizes:
            n_gram_model = self._language_models[n_gram_size]

            token_candidates = n_gram_model.generate_next_token(sequence_to_continue)

            if token_candidates is not None and len(token_candidates) > 0:
                token_probabilities = {token: freq / sum(token_candidates.values())
                                       for token, freq in token_candidates.items()}
                return token_probabilities

        return None
