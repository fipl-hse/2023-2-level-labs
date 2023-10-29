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
    if (not isinstance(raw_word, str) or
        not isinstance(start_of_word, str | None) or
        not isinstance(end_of_word, str | None)):
        return None
    word = list(raw_word)
    if start_of_word is not None:
        word.insert(0, start_of_word)
    if end_of_word is not None:
        word.insert(len(word) + 1, end_of_word)
    return tuple(word)


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
    if (not isinstance(text, str) or
        not isinstance(end_of_word, str) or
        not isinstance(start_of_word, str | None)):
        return None
    text_list = []
    for word in text.split():
        if prepare_word(word, start_of_word, end_of_word) is None:
            return None
        text_list.append(prepare_word(word, start_of_word, end_of_word))
    w_freq = {}
    for word in text_list:
        w_freq[word] = text_list.count(word)
    return w_freq


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
    pair_freq = {}
    for word in word_frequencies:
        for i in range(len(word) - 1):
            if (word[i], word[i + 1]) not in pair_freq:
                pair_freq[(word[i], word[i + 1])] = 0
            pair_freq[(word[i], word[i + 1])] += word_frequencies[word]
    return pair_freq


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if (not isinstance(word_frequencies, dict) or
        not isinstance(pair, tuple)):
        return None
    freq_2 = {}
    for word in word_frequencies:
        if ''.join(pair) in ''.join(word):
            pair_i = []
            word_list = list(word)
            for i in range(len(word) - 1):
                if (word[i], word[i + 1]) == pair:
                    pair_i.append(i)
            for i in pair_i:
                word_list[i:i + 2] = [''.join(pair)]
            freq_2[tuple(word_list)] = word_frequencies[word]
        else:
            freq_2[word] = word_frequencies[word]
    return freq_2


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if (not isinstance(word_frequencies, dict) or
        not isinstance(num_merges, int)):
        return None
    tokens_freq = count_tokens_pairs(word_frequencies)
    if tokens_freq is None:
        return None
    if num_merges > len(tokens_freq):
        num_merges = len(tokens_freq)
    m_f_pairs = [key for key, value in tokens_freq.items() if value == max(tokens_freq.values())]
    len_max = max(len(''.join(pair)) for pair in m_f_pairs)
    longest_pairs = sorted([pair for pair in m_f_pairs if len_max == len(''.join(pair))])
    word_frequencies = merge_tokens(word_frequencies, longest_pairs[0])
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
    if (not isinstance(word_frequencies, dict) or
        not isinstance(unknown_token, str)):
        return None
    tokens = set()
    tokens.add(unknown_token)
    for word in word_frequencies:
        for token in word:
            tokens.add(token)
        for symbol in ''.join(word):
            tokens.add(symbol)
    tokens = list(tokens)
    tokens_by_length = []
    max_length = max(len(token) for token in tokens)
    while max_length != 0:
        tokens_by_length.extend(sorted([token for token in tokens if len(token) == max_length]))
        max_length -= 1
    idents = {}
    for i, token in enumerate(tokens_by_length):
        idents[token] = i
    return idents


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
    if (not isinstance(encoded_text, list) or
        not isinstance(vocabulary, dict) or
        not isinstance(end_of_word_token, str | None)):
        return None
    decoded_text = ''
    for num in encoded_text:
        for token in vocabulary:
            if vocabulary[token] == num and end_of_word_token is not None:
                if token == end_of_word_token:
                    decoded_text += ' '
                else:
                    decoded_text += token
            if vocabulary[token] == num and end_of_word_token is None:
                if token == end_of_word_token:
                    decoded_text += ''
                else:
                    decoded_text += token
    if end_of_word_token is not None:
        if end_of_word_token in decoded_text:
            decoded_text.replace(end_of_word_token, ' ')
    return decoded_text


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
    if (not isinstance(word, tuple) or
        not isinstance(vocabulary, dict) or
        not isinstance(unknown_token, str) or
        not isinstance(end_of_word, str | None)):
        return None
    word_str = ''.join(word)
    tokens_by_length = []
    max_length = max(len(token) for token in vocabulary)
    while max_length != 0:
        tokens_by_length.extend(sorted([token for token in vocabulary if len(token) == max_length]))
        max_length -= 1
    word_encoded = []
    for token in tokens_by_length:
        if token in ''.join(word):
            word_str = word_str.replace(token, ' ' + str(vocabulary[token]) + ' ')
    for symbol in ''.join(word):
        if symbol not in tokens_by_length:
            word_str = word_str.replace(symbol, ' ' + str(vocabulary[unknown_token]) + ' ')
    for ind in word_str.split('  '):
        word_encoded.append(int(ind))
    return word_encoded


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None
    with open(vocab_path, 'r', encoding='utf-8') as text_file:
        vocabulary = json.load(text_file)
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
    if (not isinstance(original_text, str) or
        not isinstance(vocabulary, dict | None) or
        not isinstance(start_of_word_token, str | None) or
        not isinstance(end_of_word_token, str | None) or
        not isinstance(unknown_token, str)):
        return None
    text_list = original_text.split(' ')
    encoded_text = []
    for word in text_list:
        word_prepared = prepare_word(word, start_of_word_token, end_of_word_token)
        if word_prepared is None:
            return None
        word_encoded = tokenize_word(word_prepared, vocabulary, end_of_word_token, unknown_token)
        if word_encoded is None:
            return None
        encoded_text.extend(word_encoded)
    return encoded_text


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """
    if (not isinstance(text, str) or
        not isinstance(order, int)):
        return None
    ngrams = []
    for ind, symbol in enumerate(text):
        if ind + order > len(text):
            break
        ngrams.append(tuple(symbol for symbol in text[ind:ind + order]))
    return ngrams


def calculate_precision(
    actual: list[tuple[str, ...]], reference: list[tuple[str, ...]]
) -> float | None:
    """
    Compares two sequences by virtue of Precision metric
    :param actual: predicted sequence of n-grams
    :param reference: expected sequence of n-grams
    :return: value of Precision metric
    """
    if (not isinstance(actual, list) or
        not isinstance(reference, list)):
        return None
    if len(actual) == 0:
        return 0
    precision = 0
    reference_unique = set(reference)
    for symbol in reference_unique:
        if symbol in actual:
            precision += 1
    return precision / len(reference_unique)


def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """
    if (not isinstance(precisions, list) or
        not isinstance(max_order, int)):
        return None
    product = 1
    for order in range(max_order):
        if precisions[order] < 0:
            return 0
        product *= precisions[order]
    return product ** (1 / max_order)


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
    if (not isinstance(actual, str | None) or
        not isinstance(reference, str) or
        not isinstance(max_order, int)):
        return None
    precisions = []
    for order in range(max_order):
        actual_ngrams = collect_ngrams(actual, order + 1)
        if actual_ngrams is None:
            return None
        reference_ngrams = collect_ngrams(reference, order + 1)
        if reference_ngrams is None:
            return None
        precision = calculate_precision(actual_ngrams, reference_ngrams)
        if precision is None:
            return None
        precisions.append(precision)
    g_m = geo_mean(precisions, max_order)
    if g_m is None:
        return None
    return g_m * 100
