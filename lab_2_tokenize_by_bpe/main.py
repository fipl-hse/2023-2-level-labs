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
    if not isinstance(raw_word, str) or not (isinstance(
            start_of_word, str) or start_of_word is None) or not (
            isinstance(end_of_word, str) or end_of_word is None):
        return None
    tokenized_word = []
    if start_of_word:
        tokenized_word.append(start_of_word)
    tokenized_word.extend(raw_word)
    if end_of_word:
        tokenized_word.append(end_of_word)
    return tuple(tokenized_word)


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
    frequencies_dict = {}
    for word in text.split():
        if start_of_word is not None:
            tokenized_word = prepare_word(word, start_of_word, end_of_word)
        if start_of_word is None:
            tokenized_word = prepare_word(word, None, end_of_word)
            if tokenized_word is None:
                return None
            frequencies_dict[tokenized_word] = frequencies_dict.get(tokenized_word, 0) + 1
    return frequencies_dict


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
    pairs_of_tokens = {}
    for tokens in word_frequencies:
        for index in range(len(tokens) - 1):
            pair = (tokens[index], tokens[index + 1])
            pairs_of_tokens[pair] = pairs_of_tokens.get(pair, 0) + word_frequencies[tokens]
    return pairs_of_tokens


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not (isinstance(word_frequencies, dict)
            and isinstance(pair, tuple)):
        return None
    merged_frequencies = {}
    for preprocessed_word, count in word_frequencies.items():
        if ''.join(pair) in ''.join(preprocessed_word):
            list_word = list(preprocessed_word)
            for index in range(len(list_word) - 1):
                if (list_word[index], list_word[index + 1]) == pair:
                    list_word[index + 1] = ''.join(pair)
                    list_word[index] = ''
            if '' in list_word:
                list_word.remove('')
            preprocessed_word = tuple(list_word)
        merged_frequencies[preprocessed_word] = count
    return merged_frequencies


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
        pairs_of_tokens = count_tokens_pairs(word_frequencies)
        if not pairs_of_tokens:
            return None
        if num_merges > len(pairs_of_tokens):
            num_merges = len(pairs_of_tokens)
        pairs_max_values = ([token_pair for token_pair, frequency in pairs_of_tokens.items() if
                            frequency == max(pairs_of_tokens.values())])
        sorted_pairs = (sorted(pairs_max_values,
                               key=lambda pair: (-len(str(pair)), pair)))
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
    tokens_list = set()
    dict_token_identifier = {}
    for tuples in word_frequencies:
        for token in tuples:
            tokens_list.add(token)
            for element in token:
                tokens_list.update(element)
    tokens_list.add(unknown_token)
    sorted_tokens = sorted(tokens_list, key=lambda x: (-len(x), x))
    for index, token in enumerate(sorted_tokens):
        dict_token_identifier[token] = index
    return dict_token_identifier


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
    decoded_tokens = []
    for index in encoded_text:
        for token, token_index in vocabulary.items():
            if token_index == index and end_of_word_token is not None:
                decoded_tokens.append(' ' if token == end_of_word_token else token)
            if vocabulary[token] == index and end_of_word_token is None:
                decoded_tokens.append('' if token == end_of_word_token else token)
    decoded_text = ''.join(decoded_tokens)
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
    if (not isinstance(word, tuple) or not all(isinstance(w, str) for w in word)
            or not isinstance(vocabulary, dict) or not isinstance(
            end_of_word, (str, type(None))) or not isinstance(unknown_token, str)):
        return None
    word_str = ''.join(word)
    sorted_tokens = sorted(list(vocabulary.keys()), key=lambda x: (-len(x), x))
    for token in sorted_tokens:
        if token in ''.join(word):
            word_str = word_str.replace(token, str(vocabulary[token]) + ' ')
    for symbol in ''.join(word):
        if symbol not in sorted_tokens:
            word_str = word_str.replace(symbol, str(vocabulary[unknown_token]) + ' ')
    return [int(identifier) for identifier in word_str.split()]


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
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
    if not isinstance(original_text, str) or not isinstance(vocabulary, dict) or not isinstance(
            unknown_token, str):
        return None
    list_token_identifiers = []
    text = original_text.split()
    for word in text:
        prepared_word = prepare_word(word, start_of_word_token, end_of_word_token)
        if not prepared_word:
            return None
        tokens_id = tokenize_word(prepared_word, vocabulary, end_of_word_token, unknown_token)
        if not tokens_id:
            return None
        list_token_identifiers.extend(tokens_id)
    return list_token_identifiers


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """
    if not isinstance(text, str) or not isinstance(order, int):
        return None
    sequence_ngrams = []
    for index in range(len(text) + 1 - order):
        sequence_ngrams.append(tuple(text[index:order+index]))
    return sequence_ngrams


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
    unique_reference = set(reference)
    identical_tokens = [token for token in unique_reference if token in actual]
    precision = len(identical_tokens) / len(unique_reference)
    return precision


def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """
    if not isinstance(precisions, list) or not isinstance(max_order, int):
        return None
    if not precisions or max_order <= 0:
        return None
    all_precision = 1.0
    for precision in precisions:
        all_precision *= precision
    return float(all_precision**(1.0 / max_order))


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
    if (not isinstance(actual, str) or not isinstance(reference, str)
            or not isinstance(max_order, int)):
        return None
    all_ngrams_actual = []
    all_ngrams_reference = []
    for order in range(max_order):
        ngrams_actual = collect_ngrams(actual, order + 1)
        ngrams_reference = collect_ngrams(reference, order + 1)
        if not ngrams_actual or not ngrams_reference:
            return None
        all_ngrams_actual.append(ngrams_actual)
        all_ngrams_reference.append(ngrams_reference)
    precisions = []
    for ngrams_actual, ngrams_reference in zip(all_ngrams_actual, all_ngrams_reference):
        presision = calculate_precision(ngrams_actual, ngrams_reference)
        if not presision:
            return None
        precisions.append(presision)
    blue_metric = geo_mean(precisions, max_order)
    if blue_metric is None:
        return None
    return blue_metric * 100
