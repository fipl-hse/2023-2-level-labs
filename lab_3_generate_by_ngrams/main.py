"""
Lab 3.

Beam-search and natural language generation evaluation
"""
import json
import math
import string
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
        if isinstance(end_of_word_token, str):
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
        if not isinstance(text, str):
            return None
        words = text.split()
        tokens = []

        for word in words:
            tokenized_word = [token.lower() for token in word if token.isalpha()]
            if tokenized_word:
                tokens.extend(tokenized_word)
                tokens.append(self._end_of_word_token)
        if not tokens:
            return None
        if text[-1] not in string.punctuation and text[-1] != ' ' and text[-1] != '\n':
            tokens.pop(-1)
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
        if (
            not isinstance(element, str) or
            element not in self._storage
        ):
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
        if (
            not isinstance(element_id, int) or
            element_id not in self._storage.values()
        ):
            return None
        token = list(filter(lambda x: self._storage[x] == element_id, self._storage)).pop()
        return token

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
        if (
            not isinstance(text, str) or
            not text
        ):
            return None
        tokens = self._tokenize(text)
        if not tokens:
            return None

        encoded_corpus = []
        for token in tokens:
            self._put(token)
            identifier = self.get_id(token)
            if identifier is None:
                return None
            encoded_corpus.append(identifier)
        return tuple(encoded_corpus)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if (
            not isinstance(element, str) or
            len(element) != 1
        ):
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
        if (
                not isinstance(encoded_corpus, tuple) or
                not encoded_corpus
        ):
            return None
        decoded_corpus = self._decode(encoded_corpus)
        if not decoded_corpus:
            return None
        decoded_text = self._postprocess_decoded_text(decoded_corpus)

        return decoded_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not(isinstance(content, dict) and content):
            return None
        n_grams = list(content['freq'])
        n_grams = ''.join(n_grams).replace(' ', self._end_of_word_token).lower()
        n_grams = [x for x in n_grams if x.isalpha()]
        for n_gram in n_grams:
            self._put(n_gram)
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
        if (
                not isinstance(corpus, tuple) or
                not corpus
        ):
            return None
        decoded_corpus = []
        for symbol in corpus:
            letter = self.get_token(symbol)
            if not letter:
                return None
            decoded_corpus.append(letter)

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
        if (
                not isinstance(decoded_corpus, tuple) or
                not decoded_corpus
        ):
            return None
        result = ''.join(decoded_corpus).capitalize().replace(self._end_of_word_token, ' ')
        if result[-1] == ' ':
            result = result[:-1]
        return f"{result}."

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
        if not (isinstance(frequencies, dict) and frequencies):
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
        if not(
            isinstance(self._encoded_corpus, tuple) and self._encoded_corpus
        ):
            return 1
        n_grams = self._extract_n_grams(self._encoded_corpus)
        if not n_grams:
            return 1
        for n_gram in n_grams:
            if n_gram in self._n_gram_frequencies:
                continue
            count_current = n_grams.count(n_gram)
            cut_ngram = n_gram[:self._n_gram_size-1]
            starts_from_n_gram = len([i for i in n_grams if i[:self._n_gram_size-1] == cut_ngram])
            self._n_gram_frequencies[n_gram] = count_current / starts_from_n_gram

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
        if (
            not(isinstance(sequence, tuple) and sequence)
            or len(sequence) < self._n_gram_size - 1
        ):
            return None

        context = sequence[-(self._n_gram_size-1):]
        next_token_freq = {}
        for n_gram, frequency in self._n_gram_frequencies.items():
            if n_gram[:-1] == context:
                token = n_gram[-1]
                next_token_freq[token] = frequency

        return next_token_freq

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
        if (
            not isinstance(encoded_corpus, tuple) or
            not encoded_corpus
        ):
            return None

        ngrams = ([tuple(encoded_corpus[i:i + self._n_gram_size])
                   for i in range(len(encoded_corpus) - self._n_gram_size + 1)])
        return tuple(ngrams)

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
        if not(
            isinstance(seq_len, int) and
            isinstance(prompt, str) and prompt
        ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        n_gram_size = self._model.get_n_gram_size()
        if not encoded_prompt or not n_gram_size:
            return None
        for i in range(seq_len):
            tokens = self._model.generate_next_token(encoded_prompt)
            if not tokens:
                break
            max_freq = max(tokens.values())
            frequent_tokens = [token for token in tokens if tokens[token] == max_freq]
            token = sorted(frequent_tokens)[0]
            encoded_prompt += (token,)
        decoded = self._text_processor.decode(encoded_prompt)
        return decoded


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
        if not(isinstance(sequence, tuple) and sequence):
            return None
        variants = self._model.generate_next_token(sequence)
        if variants is None:
            return None
        if not variants:
            return []
        sorted_variants = sorted(variants.items(), key=lambda x: x[1], reverse=True)
        return sorted_variants[:self._beam_width]

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
        if not(
            isinstance(sequence_candidates, dict) and sequence_candidates and
            isinstance(sequence, tuple) and sequence and
            isinstance(next_tokens, list) and next_tokens and
            len(next_tokens) <= self._beam_width and
            sequence in sequence_candidates
        ):
            return None
        sequence_freq = sequence_candidates.pop(sequence)
        for (token, freq) in next_tokens:
            candidate = sequence + (token,)
            new_frequency = sequence_freq - math.log(freq)
            sequence_candidates[candidate] = new_frequency
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
        if not(isinstance(sequence_candidates, dict) and sequence_candidates):
            return None
        sequences = sequence_candidates.items()
        return dict(sorted(sequences, key=lambda x: (x[1], x[0]))[:self._beam_width])


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
        if not(
            isinstance(prompt, str) and prompt and
            isinstance(seq_len, int) and seq_len > 0
        ):
            return None

        encoded = self._text_processor.encode(prompt)
        if not encoded:
            return None
        candidates = {encoded: 0.0}

        for i in range(seq_len):
            copy_candidates = candidates.copy()
            for sequence in candidates:
                next_tokens = self._get_next_token(sequence)
                if not next_tokens:
                    return None
                next_tokens = sorted(next_tokens, key=lambda x: (x[1], x[0]))[:self._beam_width]
                continued = self.beam_searcher.continue_sequence(
                    sequence, next_tokens, copy_candidates
                )
                if not continued:
                    return self._text_processor.decode(sorted(candidates, key=lambda x: x[1])[0])
                best_n = self.beam_searcher.prune_sequence_candidates(copy_candidates)
                if not best_n:
                    return None
                candidates = best_n
        decoded = self._text_processor.decode(sorted(candidates, key=lambda pair: pair[1])[0])
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
        if not(isinstance(sequence_to_continue, tuple) and sequence_to_continue):
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
        with open(self._json_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        self._content = content
        self._text_processor = TextProcessor(eow_token)
        self._text_processor.fill_from_ngrams(content)

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
        if not(
                isinstance(n_gram_size, int) and n_gram_size
                and 2 <= n_gram_size <= 5
        ):
            return None
        needed_n_grams = {}
        all_n_grams = {}
        for n_gram in self._content['freq']:
            encoded_n_gram = []
            eow = self._text_processor.get_end_of_word_token()
            n_gram_copy = n_gram.replace(' ', eow)
            tokens = [i.lower() for i in n_gram_copy if i.isalpha() or i == eow]
            for token in tokens:
                identificator = self._text_processor.get_id(token)
                if identificator is None:
                    continue
                encoded_n_gram.append(identificator)
            # if tuple(encoded_n_gram) not in all_n_grams and encoded_n_gram:
            #     all_n_grams[tuple(encoded_n_gram)] = float(self._content['freq'][n_gram])

            if tuple(encoded_n_gram) not in all_n_grams:
                all_n_grams[tuple(encoded_n_gram)] = 0.0
            all_n_grams[tuple(encoded_n_gram)] += self._content['freq'][n_gram]

        for n_gram, freq in all_n_grams.items():
            if isinstance(n_gram, tuple) and len(n_gram) == n_gram_size:
                same_context = [context_freq for context, context_freq in all_n_grams.items()
                                if context[-n_gram_size:-1] == n_gram[-n_gram_size:-1]]
                needed_n_grams[n_gram] = freq / sum(same_context)

        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(needed_n_grams)
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
        if not(
            isinstance(seq_len, int) and
            isinstance(prompt, str) and prompt
        ):
            return None

        encoded = self._text_processor.encode(prompt)
        if not encoded:
            return None

        maximum = 0
        for i in range(seq_len):
            candidates = self._get_next_token(encoded)
            if not candidates:
                break

            maximum = max(candidates.values())
            best_candidate = list(filter(lambda x: candidates[x] == maximum, candidates))
            encoded += (best_candidate[0],)
        decoded_sequence = self._text_processor.decode(encoded)
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
        if not(isinstance(sequence_to_continue, tuple) and sequence_to_continue
                and self._language_models):
            return None
        n_gram_sizes = sorted(self._language_models.keys(), reverse=True)
        for size in n_gram_sizes:
            model = self._language_models[size]
            candidate = model.generate_next_token(sequence_to_continue)
            if candidate:
                return {token: freq / sum(candidate.values()) for token, freq in candidate.items()}
        return None