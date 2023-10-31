"""
Lab 2
BPE and machine translation evaluation
"""
import json
import math


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
    if not isinstance(raw_word, str) \
        or not (isinstance(start_of_word, str) or start_of_word is None) \
        or not (isinstance(end_of_word, str) or end_of_word is None):
        return None
    list_of_tokens = list(raw_word)

    if start_of_word:
        list_of_tokens.insert(0, start_of_word)
    if end_of_word:
        list_of_tokens.append(end_of_word)

    return tuple(list_of_tokens)

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
    if not isinstance(text, str) or not \
        isinstance(end_of_word, str) or not \
        (isinstance(start_of_word, str) or start_of_word is None):
        return None

    words = text.split()
    dict_of_freq = {}

    for word in words:
        freq = words.count(word)
        prepared_word = prepare_word(word, start_of_word, end_of_word)
        if not prepared_word:
            return None
        dict_of_freq.update({prepared_word:freq})

    return dict_of_freq

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

    freq_of_pairs = {}

    for word in word_frequencies:
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            if pair not in freq_of_pairs:
                freq_of_pairs[pair] = 0
            freq_of_pairs[pair] += word_frequencies[word]
    return freq_of_pairs


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict) \
        or not isinstance(pair, tuple):
        return None

    processed_dict = {}
    couple_of_items = pair[0] + pair[1]

    for word, freq in word_frequencies.items():
        if f"'{pair[0]}', '{pair[1]}'" in str(word):
            list_of_tokens = list(word)
            for index in range(len(list_of_tokens) - 1):
                if (word[index] + word[index + 1]) == couple_of_items:
                    list_of_tokens [index] = couple_of_items
                    list_of_tokens.pop(index + 1)
            processed_dict[tuple(list_of_tokens)] = freq
        else:
            processed_dict[word] = freq
    return processed_dict

def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict) or word_frequencies is None\
        or not isinstance(num_merges, int):
        return None

    while num_merges > 0:
        token_pairs = count_tokens_pairs(word_frequencies)
        if not token_pairs:
            return None

        max_freq_of_pairs = max(token_pairs.values())
        if num_merges > len(token_pairs):
            num_merges = len(token_pairs)

        max_freq = [key for key, value in token_pairs.items() if value == max_freq_of_pairs]
        max_len = max(len(str(pair)) for pair in max_freq)
        pair_of_max_freq_and_len = [pair for pair in max_freq if max_len == len(str(pair))]
        sorted_pair = sorted(pair_of_max_freq_and_len)
        word_frequencies = merge_tokens(word_frequencies, sorted_pair[0])
        if not word_frequencies:
            return None
        num_merges -= 1
    return word_frequencies

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

    dict_of_ident = {}
    diff_tokens = set()
    diff_tokens.add(unknown_token)

    for key in word_frequencies:
        for word in key:
            diff_tokens.add(word)
            for token in word:
                diff_tokens.add(token)

    alphabetic_sorting = sorted(diff_tokens)
    len_sorting = sorted(alphabetic_sorting, key = len, reverse = True )

    for index, token in enumerate(len_sorting):
        dict_of_ident[token] = index

    return dict_of_ident

def decode(
    encoded_text: list[int] | None, vocabulary: dict[str, int] | None, end_of_word_token: str | None
) -> str | None:
    """
    Translates encoded sequence into decoded one
    :param encoded_text: a sequence of token identifiers
    :param vocabulary: dictionary in the form of <token: identifier>
    :param end_of_word_token: an end-of-word token
    :return: decoded sequence
    """
    if not isinstance(encoded_text, list) \
        or not isinstance(vocabulary, dict) \
        or not (isinstance(end_of_word_token, str) \
        or end_of_word_token is None):
        return None

    decoded_sequence = ''
    for ident in encoded_text:
        list_for_token = [key for key, value in vocabulary.items() if value == ident]

        for token in list_for_token:
            if end_of_word_token:
                if end_of_word_token in token:
                    end_index = len(end_of_word_token)
                    decoded_sequence += end_of_word_token[end_index:] + ' '
                else:
                    decoded_sequence += token
            else:
                decoded_sequence += token
    return decoded_sequence

def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
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
    raw_word = ''.join(word)
    draft = ''.join(word)
    sorted_vocab = sorted(vocabulary.keys())
    sorted_vocab = sorted(sorted_vocab, key = len, reverse = True)

    for n_gramm in sorted_vocab:
        if n_gramm in raw_word:
            draft = draft.replace(n_gramm, str(vocabulary[n_gramm]) + ' ')

    for n_gramm in raw_word:
        if not n_gramm in sorted_vocab:
            draft = draft.replace(n_gramm, str(vocabulary[unknown_token]) + ' ')

    encoded_list = draft.split()
    encoded = [int(num) for num in encoded_list]
    return encoded


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, 'r', encoding='utf-8' ) as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        return None

    return vocabulary

def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
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
    if not isinstance(original_text, str) \
        or not isinstance(vocabulary, dict) \
        or not (isinstance(start_of_word_token, str) or start_of_word_token is None) \
        or not (isinstance(end_of_word_token, str) or end_of_word_token is None) \
        or not isinstance(unknown_token, str):
        return None

    list_of_ident = []
    splitted_text = original_text.split()
    for word in splitted_text:
        tuple_of_words = prepare_word(word, start_of_word_token, end_of_word_token)
        if not tuple_of_words:
            return None
        word_idents = tokenize_word(tuple_of_words, vocabulary, end_of_word_token, unknown_token)
        if not word_idents:
            return None
        list_of_ident += word_idents

    return list_of_ident

def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """
    if not isinstance(text, str) or not isinstance(order, int):
        return None

    list_of_n_gramms = []
    for index in range(0, len(text)):
        word_slice = text[index: index + order]
        if len(word_slice) == order:
            list_of_n_gramms.append(tuple(el for el in word_slice))

    return list_of_n_gramms

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

    matches = 0.0
    reference_set = set(reference)

    for n_gram in reference_set:
        if n_gram in actual:
            matches += 1
    value_of_precision = matches / len(reference_set)

    return value_of_precision

def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """
    if not isinstance(precisions, list) \
        or not isinstance(max_order, int):
        return None
    total_sum = 0.0

    for order in range(max_order):
        if not precisions[order] < 0:
            total_sum += math.log(precisions[order])
        else:
            return 0
    average_geo_mean = math.exp(1/max_order * total_sum)

    return average_geo_mean

def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
    if not isinstance(actual, str) or not isinstance(reference, str) \
        or max_order != 3:
        return None
    actual_ngrams = []
    expected_ngrams = []

    for order in range(max_order):
        list_for_actual_ngrams = collect_ngrams(actual, order + 1)
        list_for_expected_ngrams = collect_ngrams(reference, order + 1)
        if list_for_actual_ngrams is None or list_for_expected_ngrams is None:
            return None
        actual_ngrams.append(list_for_actual_ngrams)
        expected_ngrams.append(list_for_expected_ngrams)

    precisions = []
    for actual_ngram, expected_ngram in zip(actual_ngrams, expected_ngrams):
        precision_value = calculate_precision(actual_ngram, expected_ngram)
        if precision_value is None:
            return None
        precisions.append(precision_value)

    average_geo_mean = geo_mean(precisions, max_order)
    if average_geo_mean is None:
        return None

    return average_geo_mean * 100
