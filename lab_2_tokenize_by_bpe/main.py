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
    if not isinstance(text, str) or not isinstance(end_of_word, str) or not (
            isinstance(start_of_word, str) or start_of_word is None):
        return None

    dict_frequencies = {}

    splitted_text = text.split()
    for i in set(splitted_text):
        word = prepare_word(i, start_of_word, end_of_word)
        if not word:
            return None
        dict_frequencies[word] = splitted_text.count(i)

    return dict_frequencies

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

    processed_dict = {}
    couple_of_items = pair[0] + pair[1]

    for word, freq in word_frequencies.items():
        list_of_tokens = list(word)
        for index in range(len(list_of_tokens) - 1):
            if (word[index] + word[index + 1]) == couple_of_items:
                list_of_tokens [index] = couple_of_items
                list_of_tokens [index + 1] = ''
        if '' in list_of_tokens:
            list_of_tokens.remove('')
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
    if not isinstance(word_frequencies, dict) or not isinstance(num_merges, int):
        return None
    dict_with_pairs = count_tokens_pairs(word_frequencies)

    if not dict_with_pairs:
        return None
    merges = min(num_merges, len(dict_with_pairs))

    for i in range(merges):

        max_values = max(dict_with_pairs.values())
        pairs_max_values = [i for i in dict_with_pairs if dict_with_pairs[i] == max_values]

        max_len = max(len(str(pair)) for pair in pairs_max_values)
        pairs_max_len = [i for i in pairs_max_values if len(str(i)) == max_len]

        sorted_pairs = sorted(pairs_max_len)
        word_frequencies = merge_tokens(word_frequencies, sorted_pairs[0])

        if not word_frequencies:
            return None

        dict_with_pairs = count_tokens_pairs(word_frequencies)

        if not dict_with_pairs:
            return None

    return word_frequencies

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
    if not isinstance(word_frequencies, dict) or not isinstance(unknown_token, str):
        return None

    dict_ident = {}
    unique_tokens = set()

    for tuple_tokens in word_frequencies.keys():
        for word in tuple_tokens:
            unique_tokens.update(tuple_tokens, word)

    unique_tokens.add(unknown_token)
    lex_sorted = sorted(unique_tokens)
    len_sorted = sorted(lex_sorted, key=len, reverse=True)
    index = 0

    for token in len_sorted:
        dict_ident[token] = index
        index += 1

    return dict_ident

    dict_of_ident = {}
    diff_tokens = set()
    diff_tokens.add(unknown_token)

    for key in word_frequencies:
        for word in key:
            diff_tokens.add(word)
            diff_tokens.update(token for token in word)

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
    if not isinstance(encoded_text, list) or not isinstance(vocabulary, dict) or not (isinstance(
            end_of_word_token, str) or end_of_word_token is None):
        return None
    decoded = ''
    for identifier in encoded_text:
        token_list = [key for key in vocabulary if vocabulary[key] == identifier]

        for token in token_list:
            decoded += token

    if end_of_word_token:
        decoded = decoded.replace(end_of_word_token, ' ')

    return decoded

    decoded_sequence = ''
    for ident in encoded_text:
        list_for_token = [key for key, value in vocabulary.items() if value == ident]

        for token in list_for_token:
            decoded_sequence += token
        if end_of_word_token:
            if end_of_word_token in decoded_sequence:
                decoded_sequence = decoded_sequence.replace(end_of_word_token, ' ')

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
    if not isinstance(word, tuple) or not isinstance(vocabulary, dict) or not (isinstance(
            end_of_word, str) or end_of_word is None) or not isinstance(unknown_token, str):
        return None

    word_copy = ''.join(word)
    sorted_vocabulary = sorted(list(vocabulary.keys()), key=lambda x: (-len(x), x))
    result = []

    for key in sorted_vocabulary:
        while key in word_copy:
            index = word_copy.count(' ', 0, word_copy.find(key))
            result.insert(index, vocabulary[key])
            word_copy = word_copy.replace(key, ' ', 1)

    for unk in word_copy:
        if unk != ' ':
            index = word_copy.find(unk)
            word_copy = word_copy.replace(unk, ' ')
            result.insert(index, vocabulary[unknown_token])

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
    if not isinstance(original_text, str) or not isinstance(
            vocabulary, dict) or not (isinstance(
            start_of_word_token, str) or start_of_word_token is None) or not (isinstance(
            end_of_word_token, str) or end_of_word_token is None) or not isinstance(
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

    n_grams = []
    for index in range(len(text) + 1 - order):
        n_grams.append(tuple(text[index: index + order]))

    return n_grams

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

    unique_ngrams = set(reference)
    matches = 0

    for n_gram in unique_ngrams:
        if n_gram in actual:
            matches += 1

    return matches / len(unique_ngrams)

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
    if not isinstance(precisions, list) or not isinstance(max_order, int):
        return None

    summation = float(0)

    for order in range(max_order):
        if precisions[order] < 0:
            return 0
        summation += math.log(precisions[order])

    return math.exp(1 / max_order * summation)

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
    if not isinstance(actual, str) or not isinstance(
            reference, str) or max_order != 3:
        return None

    actual_ngrams = []
    reference_ngrams = []

    for order in range(max_order):
        actual_ngram = collect_ngrams(actual, order + 1)
        reference_ngram = collect_ngrams(reference, order + 1)
        if actual_ngram is None or reference_ngram is None:
            return None
        actual_ngrams.append(actual_ngram)
        reference_ngrams.append(reference_ngram)

    precisions = []

    for i, j in zip(actual_ngrams, reference_ngrams):
        precision = calculate_precision(i, j)
        if precision is None:
            return None
        precisions.append(precision)

    average = geo_mean(precisions, max_order)
    if average is None:
        return None

    return average * 100
