"""
Lab 2
BPE and machine translation evaluation
"""
import json


def prepare_word(
        raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenizes word into unigrams and appends end-of-word token
    :param raw_word: original word
    :param start_of_word: a token that signifies the start of word
    :param end_of_word: a token that signifies the end of word
    :return: preprocessed word
    """
    if not isinstance(raw_word, str) or \
            not (isinstance(start_of_word, str) or start_of_word is None) or \
            not (isinstance(end_of_word, str) or end_of_word is None):
        return None

    if not start_of_word and not end_of_word:
        return tuple(list(raw_word))
    if not start_of_word:
        return tuple(list(raw_word) + [str(end_of_word)])
    if not end_of_word:
        return tuple([start_of_word] + list(raw_word))
    return tuple([start_of_word] + list(raw_word) + [end_of_word])


def collect_frequencies(
        text: str, start_of_word: str | None, end_of_word: str
) -> dict[tuple[str, ...], int] | None:
    """
    Counts number of occurrences of each word
    :param text: original text with no preprocessing
    :param start_of_word: a token that signifies the start of word
    :param end_of_word: a token that signifies the end of word
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(text, str) or not isinstance(end_of_word, str) or \
            not (isinstance(start_of_word, str) or start_of_word is None):
        return None

    words = text.split()
    prepared_words = [prepare_word(word, start_of_word, end_of_word) for word in words]

    freq = {}
    for word in set(prepared_words):
        if word is None:
            return None
        freq[word] = prepared_words.count(word)

    return freq


def count_tokens_pairs(
        word_frequencies: dict[tuple[str, ...], int]
) -> dict[tuple[str, str], int] | None:
    """
    Counts number of occurrences of each pair of subsequent tokens
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :return: dictionary in the form of <token pair: number of occurrences>
    """
    if not isinstance(word_frequencies, dict):
        return None

    seq_freq = {}
    for key in word_frequencies:
        for token_index in range(len(key) - 1):
            tokens = (key[token_index], key[token_index + 1])
            if tokens not in seq_freq:
                seq_freq[tokens] = 0
            seq_freq[tokens] += word_frequencies[key]

    return seq_freq


def merge_tokens(
        word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict) or \
            not isinstance(pair, tuple):
        return None

    preprocessed_freq = {}

    for key, frequency in word_frequencies.items():
        if str(pair)[1:-2] in str(key):
            pair_indexes = []
            for token_index in range(len(key) - 1):
                tokens = (key[token_index], key[token_index + 1])
                if tokens == pair:
                    pair_indexes.append(token_index)

            saved_tokens = list(key)
            for index in pair_indexes:
                saved_tokens[index:index + 2] = [''.join(pair)]

            preprocessed_freq[tuple(saved_tokens)] = frequency
        else:
            preprocessed_freq[key] = frequency

    return preprocessed_freq


def train(
        word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict) or not isinstance(num_merges, int):
        return None

    token_pairs = count_tokens_pairs(word_frequencies)
    if not token_pairs:
        return None
    num_merges = min(num_merges, len(token_pairs))

    occur_max = max(token_pairs.values())
    pair_max_occur = [key for key, value in token_pairs.items() if value == occur_max]

    len_max = max(len(''.join(pair)) for pair in pair_max_occur)
    pair_max_len = [pair for pair in pair_max_occur if len_max == len(''.join(pair))]

    preferred_pairs = sorted(pair_max_len)

    word_frequencies = merge_tokens(word_frequencies, preferred_pairs[0])
    if not word_frequencies:
        return None
    if num_merges == 1:
        return word_frequencies
    return train(word_frequencies, num_merges - 1)


def get_vocabulary(
        word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establishes correspondence between tokens and its integer identifier
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param unknown_token: a token to signify an unknown token
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(word_frequencies, dict) \
            or not isinstance(unknown_token, str):
        return None

    words = [unknown_token]

    for tokens in word_frequencies:
        for token in tokens:
            words.append(token)
            tokenized_word = prepare_word(token, None, None)
            if not tokenized_word:
                return None
            words += list(tokenized_word)

    unique_words = set(words)
    alphabetical_order = sorted(unique_words)
    len_alp_sorted = sorted(alphabetical_order, key=len, reverse=True)

    identifiers = {}
    for index, word in enumerate(len_alp_sorted):
        identifiers[word] = index

    return identifiers


def decode(
        encoded_text: list[int], vocabulary: dict[str, int] | None, end_of_word_token: str | None
) -> str | None:
    """
    Translates encoded sequence into decoded one
    :param encoded_text: a sequence of token identifiers
    :param vocabulary: dictionary in the form of <token: identifier>
    :param end_of_word_token: an end-of-word token
    :return: decoded sequence
    """
    if not isinstance(encoded_text, list) \
            or not all(isinstance(el, int) for el in encoded_text) \
            or not isinstance(vocabulary, dict) \
            or not (isinstance(end_of_word_token, str) or end_of_word_token is None):
        return None

    encoded_dict = {}
    for index, num in enumerate(encoded_text):
        if num not in encoded_dict:
            encoded_dict[num] = []
        encoded_dict[num].append(index)

    decoded_text = [''] * len(encoded_text)

    for token, identifier in vocabulary.items():
        if end_of_word_token and end_of_word_token in token and identifier in encoded_dict:
            space_indexes = encoded_dict[identifier]
            for space_index in space_indexes:
                decoded_text[space_index] = token[:-4] + ' '
        elif identifier in encoded_dict:
            indexes = encoded_dict[identifier]
            for index in indexes:
                decoded_text[index] = token

    return ''.join(decoded_text)


def tokenize_word(
        word: tuple[str, ...], vocabulary: dict[str, int],
        end_of_word: str | None, unknown_token: str
) -> list[int] | None:
    """
    Splits word into tokens
    :param word: preprocessed word
    :param vocabulary: dictionary in the form of <token: identifier>
    :param end_of_word: an end-of-word token
    :param unknown_token: token that signifies unknown sequence
    :return: list of token identifiers
    """
    if not isinstance(word, tuple) or not isinstance(vocabulary, dict) \
            or not (isinstance(end_of_word, str) or end_of_word is None) \
            or not isinstance(unknown_token, str):
        return None

    if end_of_word:
        tokens = sorted(list(vocabulary) + [end_of_word], key=len, reverse=True)
    else:
        tokens = sorted(list(vocabulary), key=len, reverse=True)

    word_str = ''.join([str(el) for el in word])
    word_list = list(word)

    for token in tokens:
        if token not in word_str:
            continue
        if token in word_str and not token.isdigit():
            length = len(word)
            for start_index in range(length):
                for end_index in range(start_index, length):
                    potential_token = ''.join(word_list[start_index:end_index + 1])
                    if potential_token == token:
                        word_list[start_index: end_index + 1] = [str(vocabulary[token])]

            if ''.join(word_list).isdigit():
                return [int(num) for num in word_list]
            return tokenize_word(tuple(word_list), vocabulary, end_of_word, unknown_token)

    for index, num in enumerate(word_list):
        if not num.isdigit():
            word_list[index] = str(vocabulary[unknown_token])
    return [int(num) for num in word_list]


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    if not vocabulary:
        return None
    return dict(vocabulary)


def encode(
        original_text: str,
        vocabulary: dict[str, int],
        start_of_word_token: str | None,
        end_of_word_token: str | None,
        unknown_token: str,
) -> list[int] | None:
    """
    Translates decoded sequence into encoded one
    :param original_text: original text
    :param vocabulary: dictionary in the form of <token: identifier>
    :param start_of_word_token: a start-of-word token
    :param end_of_word_token: an end-of-word token
    :param unknown_token: token that signifies unknown sequence
    :return: list of token identifiers
    """
    if not isinstance(original_text, str) or not isinstance(vocabulary, dict) \
            or not (isinstance(start_of_word_token, str) or start_of_word_token is None) \
            or not (isinstance(end_of_word_token, str) or end_of_word_token is None) \
            or not isinstance(unknown_token, str):
        return None

    encoded = []
    text_words = original_text.split()
    for word in text_words:
        prepared_word = prepare_word(word, start_of_word_token, end_of_word_token)
        if not prepared_word:
            return None
        tokens = tokenize_word(prepared_word, vocabulary, end_of_word_token, unknown_token)
        if not tokens:
            return None
        encoded.extend(tokens)

    return encoded


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """


def calculate_precision(
        actual: list[tuple[str, ...]], reference: list[tuple[str, ...]]
) -> float | None:
    """
    Compares two sequences by virtue of Precision metric
    :param actual: predicted sequence of n-grams
    :param reference: expected sequence of n-grams
    :return: value of Precision metric
    """
    if not isinstance(actual, list) or not isinstance(reference, list):
        return None
    if len(actual) == 0:
        return 0.0
    true_positive = 0
    for token in set(reference):
        if token in actual:
            true_positive += actual.count(token)
    return float(true_positive / len(reference))


def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
