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
        if not isinstance(text, str) or not text:
            return None

        tokens = []
        for token in text.lower():
            if token.isspace() and tokens[-1] != self._end_of_word_token:
                tokens.append(self._end_of_word_token)
            elif token.isalpha():
                tokens.append(token)
        if not text[-1].isalnum() and tokens[-1] != self._end_of_word_token:
            tokens.append(self._end_of_word_token)

        if tokens.count(self._end_of_word_token) == len(tokens):
            return None
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
        for key, identificator in self._storage.items():
            if identificator == element_id:
                return key
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
        if not isinstance(text, str) or not text:
            return None
        encoded = []
        tokens = self._tokenize(text)
        if not tokens:
            return None
        for token in tokens:
            self._put(token)
            identificator = self.get_id(token)
            if not isinstance(identificator, int):
                return None
            encoded.append(identificator)
        return tuple(encoded)

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
        decoded_symbols = self._decode(encoded_corpus)
        if not decoded_symbols:
            return None
        decoded_text = self._postprocess_decoded_text(decoded_symbols)
        if not decoded_text:
            return None
        return decoded_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if isinstance(content, dict) and content:
            for key in content['freq']:
                for element in key:
                    if element.isalpha():
                        self._put(element.lower())

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
        tokens = []
        for identificator in corpus:
            if not isinstance(identificator, int):
                return None
            token = self.get_token(identificator)
            if not token:
                return None
            tokens.append(token)
        return tuple(tokens)

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
        if decoded_corpus[-1] != self._end_of_word_token:
            return f"{decoded_corpus[0].upper()}{''.join(decoded_corpus)[1:]}."\
                .replace(self._end_of_word_token, ' ')
        return f"{decoded_corpus[0].upper()}{''.join(decoded_corpus)[1:-1]}."\
                .replace(self._end_of_word_token, ' ')


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
        self._n_gram_size = n_gram_size
        self._n_gram_frequencies = {}
        self._encoded_corpus = encoded_corpus

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
        if not (isinstance(n_grams, tuple) and n_grams):
            return 1

        contexts = {}
        for n_gram in n_grams:
            if not isinstance(n_gram, tuple):
                return 1
            if n_gram[:-1] not in contexts:
                contexts[n_gram[:-1]] = {}
            if n_gram not in contexts[n_gram[:-1]]:
                contexts[n_gram[:-1]][n_gram] = 0.0
            contexts[n_gram[:-1]][n_gram] += 1.

        for same_context_ngrams in contexts.values():
            same_context_count = sum(same_context_ngrams.values())
            for n_gram, freq in same_context_ngrams.items():
                self._n_gram_frequencies[n_gram] = freq/same_context_count
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
        if not (isinstance(sequence, tuple) and sequence
                and len(sequence) >= self._n_gram_size - 1):
            return None
        context = sequence[-self._n_gram_size + 1:]
        return {n_gram[-1]: freq
                for n_gram, freq in self._n_gram_frequencies.items()
                if n_gram[:-1] == context}

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
        if not (isinstance(seq_len, int) and isinstance(prompt, str) and prompt):
            return None
        n_gram_size = self._model.get_n_gram_size()
        encoded = self._text_processor.encode(prompt)
        if not (encoded and n_gram_size):
            return None

        phrase = prompt
        for iteration in range(seq_len):
            tokens = self._model.generate_next_token(encoded[-n_gram_size + 1:])
            if not tokens:
                break
            max_freq = max(tokens.values())
            candidates_max = [candidate for candidate, freq in tokens.items()
                              if freq == max_freq]
            encoded = encoded + (sorted(candidates_max)[0],)
            best_candidate = self._text_processor.get_token(encoded[-1])
            if not best_candidate:
                return None
            phrase += best_candidate

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
        tokens = self._model.generate_next_token(sequence)
        if tokens is None:
            return None
        if not tokens:
            return []
        return sorted([(token, float(freq)) for token, freq in tokens.items()],
                      key=lambda pair: pair[1], reverse=True)[:self._beam_width]

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
            sequence_candidates (dict[tuple[int, ...], float]): Storage with all sequences generated

        Returns:
            Optional[dict[tuple[int, ...], float]]: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if not (isinstance(sequence, tuple) and isinstance(next_tokens, list)
                and isinstance(sequence_candidates, dict) and sequence
                and next_tokens and sequence_candidates and len(next_tokens) <= self._beam_width
                and sequence in sequence_candidates):
            return None

        for (token, freq) in next_tokens:
            sequence_candidates[sequence + (token,)] = \
                sequence_candidates[sequence] - math.log(freq)
        sequence_candidates.pop(sequence)
        return sequence_candidates

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> Optional[dict[tuple[int, ...], float]]:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float]: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if not (isinstance(sequence_candidates, dict) and sequence_candidates):
            return None
        return dict(sorted(list(sequence_candidates.items()),
                           key=lambda pair: pair[1])[:self._beam_width])


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (tuple[NGramLanguageModel]): Language models for next token prediction
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
        _beam_width (int): Beam width parameter for generation
        beam_searcher (BeamSearcher): Searcher instances for each language model
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
        if not (isinstance(prompt, str) and isinstance(seq_len, int) and prompt and seq_len):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None
        seq_candidates = {encoded_prompt: 0.0}

        for iteration in range(seq_len):
            new_sequences = dict(seq_candidates)
            for sequence in seq_candidates:
                possible_tokens = self._get_next_token(sequence)
                if not possible_tokens:
                    return None
                possible_sequences = self.beam_searcher.continue_sequence(sequence,
                                                                          possible_tokens,
                                                                          new_sequences)
                if not possible_sequences:
                    return self._text_processor.decode(sorted(tuple(seq_candidates),
                                                              key=lambda pair: pair[1])[0])

            best_sequences = self.beam_searcher.prune_sequence_candidates(new_sequences)
            if not best_sequences:
                return None
            seq_candidates = best_sequences

        decoded_sequence = self._text_processor.decode(sorted(tuple(seq_candidates),
                                                              key=lambda pair: pair[1])[0])
        return decoded_sequence

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
        _content (dict): ngrams from external JSON
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
            text = json.load(file)
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
        if not (isinstance(n_gram_size, int) and 5 >= n_gram_size >= 2):
            return None

        n_grams = {}
        for key in self._content['freq']:
            encoded = []
            for token in key:
                if token.isspace():
                    encoded.append(0)
                elif token.isalpha():
                    ident = self._text_processor.get_id(token.lower())
                    if not ident:
                        continue
                    encoded.append(ident)

            if tuple(encoded) not in n_grams:
                n_grams[tuple(encoded)] = 0.0
            n_grams[tuple(encoded)] += self._content['freq'][key]

        correct_size_ngrams = {}
        for n_gram, freq in n_grams.items():
            if isinstance(n_gram, tuple) and len(n_gram) == n_gram_size:
                same_context = [context_freq for context, context_freq in n_grams.items()
                                    if context[-n_gram_size:-1] == n_gram[-n_gram_size:-1]]
                correct_size_ngrams[n_gram] = freq / sum(same_context)

        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(correct_size_ngrams)
        return model

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
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
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
        self._language_models = {model.get_n_gram_size(): model for model in
                                 sorted(language_models,
                                        key=lambda model: model.get_n_gram_size(),
                                        reverse=True)}
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
        if not(isinstance(seq_len, int) and isinstance(prompt, str) and prompt):
            return None

        encoded_sequence = self._text_processor.encode(prompt)
        if not encoded_sequence:
            return None
        for iteration in range(seq_len):
            candidates = self._get_next_token(encoded_sequence)
            if not candidates:
                break
            max_probability = max(candidates.values())
            best_candidate = [token for token, freq in candidates.items()
                              if freq == max_probability]
            encoded_sequence += (best_candidate[0],)
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
        if not(isinstance(sequence_to_continue, tuple) and sequence_to_continue and
               self._language_models):
            return None

        candidates = {}
        best_size = 0
        for n_gram_size, model in self._language_models.items():
            if n_gram_size < best_size:
                break
            for size in range(n_gram_size, 1, -1):
                model_candidates = model.generate_next_token(sequence_to_continue)
                if not model_candidates:
                    continue
                candidates.update(model_candidates)
                best_size = size
                break
        if not candidates:
            return None
        return candidates
