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
    if not (
            isinstance(raw_word, str)
            and isinstance(start_of_word, str | None)
            and isinstance(end_of_word, str | None)
    ):
        return None

    tuple_1 = list(raw_word)

    if start_of_word is not None:
        tuple_1.insert(0, start_of_word)

    if end_of_word is not None:
        tuple_1.append(end_of_word)

    return tuple(tuple_1)


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
    if not (
            isinstance(text, str)
            and isinstance(start_of_word, str | None)
            and isinstance(end_of_word, str)
    ):
        return None

    collection = text.split()
    freq = {}

    for i in set(collection):
        prepared_word = prepare_word(i, start_of_word, end_of_word)

        if prepared_word is None:
            return None

        freq[prepared_word] = collection.count(i)

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

    tokens_pairs = {}

    for word in word_frequencies:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            if pair in tokens_pairs:
                tokens_pairs[pair] += word_frequencies[word]
            else:
                tokens_pairs[pair] = word_frequencies[word]

    return tokens_pairs


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not (
            isinstance(word_frequencies, dict)
            and isinstance(pair, tuple)
    ):
        return None

    new_word_frequencies = {}

    for word in word_frequencies:
        if pair[0] in word and pair[1] in word:
            new_word = []

            for i, token in enumerate(word):
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])

                elif not (word[i - 1] == pair[0] and word[i] == pair[1]):
                    new_word.append(word[i])

            new_word_frequencies[tuple(new_word)] = word_frequencies[word]

        else:
            new_word_frequencies[word] = word_frequencies[word]

    return new_word_frequencies


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not (
            isinstance(word_frequencies, dict)
            and isinstance(num_merges, int)
    ):
        return None

    token_pairs = count_tokens_pairs(word_frequencies)

    if token_pairs is None:
        return None

    if num_merges > len(token_pairs):
        num_merges = len(token_pairs)

    for _ in range(num_merges):
        token_pairs = count_tokens_pairs(word_frequencies)
        if not token_pairs:
            return None
        max_result = max(token_pairs.values())
        max_results = {k: v for k, v in token_pairs.items() if v == max_result}

        if len(max_results) == 1:
            word_frequencies = merge_tokens(word_frequencies, list(max_results.keys())[0])
        elif len(max_results) > 1:
            key_sorted = sorted(max_results.keys(), key=lambda x: (-len(''.join(x)), x))
            word_frequencies = merge_tokens(word_frequencies, key_sorted[0])

        if not word_frequencies:
            return None

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
    if not (
            isinstance(word_frequencies, dict)
            and isinstance(unknown_token, str)
    ):
        return None

    all_tokens = [unknown_token]
    for word in word_frequencies.keys():
        for i in word:
            all_tokens.append(i)
            prepared_word = prepare_word(i, None, None)

            if prepared_word is None:
                return None

            all_tokens += list(prepared_word)

    unic_tokens = set(all_tokens)
    all_tokens_sort = sorted(unic_tokens, key=lambda x: (-len(x), x))
    int_arr = {i: num_id for num_id, i in enumerate(all_tokens_sort)}

    return int_arr


def decode(
    encoded_text: list[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None
) -> str | None:
    """
    Translates encoded sequence into decoded one
    :param encoded_text: a sequence of token identifiers
    :param vocabulary: dictionary in the form of <token: identifier>
    :param end_of_word_token: an end-of-word token
    :return: decoded sequence
    """
    if not (
            isinstance(encoded_text, list) and all(isinstance(numb, int) for numb in encoded_text)
            and isinstance(vocabulary, dict)
            and isinstance(end_of_word_token, str | None)
    ):
        return None

    vocab_reversed = {v: k for k, v in vocabulary.items()}
    decoded_text = ''
    for i in encoded_text:
        token = vocab_reversed[i]
        decoded_text += token

    if end_of_word_token is not None:
        decoded_text = decoded_text.replace(end_of_word_token, ' ')

    return decoded_text


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

    return matches / len(unique_ngrams)


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
