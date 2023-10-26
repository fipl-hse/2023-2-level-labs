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
    if not isinstance(raw_word, str) or not (isinstance(
            start_of_word, str) or start_of_word is None) or not (
            isinstance(end_of_word, str) or end_of_word is None):
        return None
    list_of_tokens = list(raw_word)
    if end_of_word:
        list_of_tokens.append(end_of_word)
    if start_of_word:
        list_of_tokens.insert(0, start_of_word)
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
    if not isinstance(text, str) or not isinstance(end_of_word, str) or not (
            isinstance(start_of_word, str) or start_of_word is None):
        return None
    dict_frequencies = {}
    for i in text.split():
        word = prepare_word(i, start_of_word, end_of_word)
        if not word:
            return None
        dict_frequencies[word] = text.split().count(i)
    return dict_frequencies


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
    dict_with_pairs = {}
    for word in word_frequencies:
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            if pair not in dict_with_pairs:
                dict_with_pairs[pair] = 0
            dict_with_pairs[pair] += word_frequencies[word]
    return dict_with_pairs


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict) or not isinstance(pair, tuple):
        return None
    dict_merged_tokens = {}
    for i in word_frequencies:
        if f"'{pair[0]}', '{pair[1]}'" in str(i):
            list_word = list(i)

            for index in range(len(list_word) - 1):
                if (i[index], i[index + 1]) == pair:
                    list_word[index + 1] = pair[0] + pair[1]
                    list_word[index] = ''

            if '' in list_word:
                list_word.remove('')

            dict_merged_tokens.update({tuple(list_word): word_frequencies[i]})
        else:
            dict_merged_tokens.update({i: word_frequencies[i]})
    return dict_merged_tokens


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
    while num_merges > 0:
        dict_with_pairs = count_tokens_pairs(word_frequencies)

        if not dict_with_pairs:
            return None
        if num_merges > len(dict_with_pairs):
            num_merges = len(dict_with_pairs)

        max_values = max(dict_with_pairs.values())
        pairs_max_values = [i for i in dict_with_pairs if dict_with_pairs[i] == max_values]

        max_len = max(len(str(pair)) for pair in pairs_max_values)
        pairs_max_len = [i for i in pairs_max_values if len(str(i)) == max_len]

        sorted_pairs = sorted(pairs_max_len)
        word_frequencies = merge_tokens(word_frequencies, sorted_pairs[0])

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
    if not isinstance(word_frequencies, dict) or not isinstance(unknown_token, str):
        return None

    dict_ident = {}
    unique_tokens = set()

    for tuple_tokens in word_frequencies.keys():
        for word in tuple_tokens:
            unique_tokens.add(word)
            for token in word:
                unique_tokens.add(token)

    unique_tokens.add(unknown_token)
    lex_sorted = sorted(unique_tokens)
    len_sorted = sorted(lex_sorted, key=len, reverse=True)
    index = 0

    for token in len_sorted:
        dict_ident[token] = index
        index += 1

    return dict_ident


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
    if not isinstance(encoded_text, list) or not isinstance(vocabulary, dict) or not (isinstance(
            end_of_word_token, str) or end_of_word_token is None):
        return None
    decoded = ''
    for identifier in encoded_text:
        token_list = [key for key in vocabulary if vocabulary[key] == identifier]

        for token in token_list:
            if end_of_word_token:
                if end_of_word_token in token:
                    decoded += token[:-4] + ' '
                else:
                    decoded += token
            else:
                decoded += token

    return decoded


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
    if not isinstance(word, tuple) or not isinstance(vocabulary, dict) or not isinstance(
            end_of_word, str) or not isinstance(unknown_token, str):
        return None

    str_word = ''.join(word)
    word_copy = ''.join(word)
    sorted_vocabulary = sorted(list(vocabulary.keys()), key=lambda x: (-len(x), x))
    result = []

    for key in sorted_vocabulary:
        if key in str_word:
            word_copy = word_copy.replace(key, ' ' + str(vocabulary[key]) + ' ')

    for unk in str_word:
        if unk not in sorted_vocabulary:
            word_copy = word_copy.replace(unk, ' ' + str(vocabulary[unknown_token]) + ' ')

    str_result = word_copy.split('  ')
    print(word_copy)

    for number in str_result:
        result.append(int(number))
    return result


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict):
        return None

    return vocab


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
    if not isinstance(original_text, str) or not isinstance(vocabulary, dict) or not isinstance(
            unknown_token, str):
        return None

    encoded = []
    split_text = original_text.split()

    for word in split_text:
        prepared = prepare_word(word, start_of_word_token, end_of_word_token)
        if not prepared:
            return None
        result = tokenize_word(prepared, vocabulary, end_of_word_token, unknown_token)
        if not result:
            return None
        encoded.extend(result)

    return encoded


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """
    if not isinstance(text, str) or not isinstance(order, int):
        return None

    n_grams = []
    for index in range(len(text) + 1 - order):
        n_grams.append(tuple(text[index: index + order]))

    return n_grams


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

    unique_ngrams = set(reference)
    matches = 0

    for n_gram in unique_ngrams:
        if n_gram in actual:
            matches += 1

    return matches/len(unique_ngrams)


def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """
    if not isinstance(precisions, list) or not isinstance(max_order,int):
        return None

    summation = float(0)

    for order in range(max_order):
        if precisions[order] < 0:
            return 0
        summation += math.log(precisions[order])

    return math.exp(1/max_order*summation)


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
