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
    if (not isinstance(raw_word, str)
       or not isinstance(start_of_word, str | None)
       or not isinstance(end_of_word, str | None)):
        return None
    tokens = list(symbol for symbol in raw_word)
    if start_of_word:
        tokens.insert(0, start_of_word)
    if end_of_word:
        tokens.append(end_of_word)
    return tuple(tokens)


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
    if (not isinstance(text, str)
       or not isinstance(start_of_word, str | None)
       or not isinstance(end_of_word, str)):
        return None
    words = text.split()
    freq_dict = {prepare_word(word, start_of_word, end_of_word): words.count(word) for word in set(words)}
    if None in freq_dict:
        return None
    return freq_dict


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
    pair_dict = {}
    for word in word_frequencies:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in pair_dict:
                pair_dict[pair] = 0
            pair_dict[pair] += word_frequencies[word]
    return pair_dict


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if (not isinstance(word_frequencies, dict)
       or not isinstance(pair, tuple)):
        return None
    new_dict = {}
    for word in word_frequencies:
        new_word = []
        ''' Alternative variation with a switch:
        second_merged = False
        for i in range(len(word)):
            if second_merged:
                second_merged = False
                continue
            if word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                second_merged = True
            else:
                new_word.append(word[i])'''
        for i in range(len(word)):
            if word[i - 1] == pair[0] and word[i] == pair[1]:
                del new_word[-1]
                new_word.append(pair[0] + pair[1])
            else:
                new_word.append(word[i])
        new_dict[tuple(new_word)] = word_frequencies[word]
    return new_dict


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if (not isinstance(word_frequencies, dict | None)
       or not isinstance(num_merges, int)):
        return None
    merged_text = word_frequencies
    for n in range(num_merges):
        tokens = count_tokens_pairs(merged_text)
        if not tokens:
            break
        top_tokens = sorted(tokens.items(), key=lambda x: (-x[1], -len(''.join(x[0])), str(x[0]).lower()))
        merged_text = merge_tokens(merged_text, top_tokens[0][0])
        if not merged_text:
            return None
    return merged_text


def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establishes correspondence between tokens and its integer identifier
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param unknown_token: a token to signify an unknown token
    :return: dictionary in the form of <token: identifier>
    """
    if (not isinstance(word_frequencies, dict)
            or not isinstance(unknown_token, str)):
        return None
    tokens = [unknown_token]
    for word in word_frequencies:
        for word_part in word:
            tokens.append(word_part)
            tokens.extend(word_part)
    unique_tokens = set(tokens)
    sorted_tokens = sorted(unique_tokens, key=lambda x: (-len(x), str(x[0])))
    return {token: i for i, token in enumerate(sorted_tokens)}


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
    if (not isinstance(encoded_text, list)
       or not isinstance(vocabulary, dict)
       or not isinstance(end_of_word_token, str | None)):
        return None
    decoded_text = ''
    reverse_vocabulary = {pair[1]: (pair[0].replace(end_of_word_token, ' ') if end_of_word_token else pair[0])
                          for pair in vocabulary.items()}
    for code in encoded_text:
        decoded_text += reverse_vocabulary[code]
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
    if (not isinstance(word, tuple)
       or not isinstance(vocabulary, dict)
       or not isinstance(end_of_word, str | None)
       or not isinstance(unknown_token, str)):
        return None
    to_be_found = ''.join(word)
    encoded = []
    for token in vocabulary:
        while token in to_be_found:
            position = to_be_found.count(' ', 0, to_be_found.find(token))
            encoded.insert(position, vocabulary[token])
            to_be_found = to_be_found.replace(token, ' ', 1)
    for i, symbol in enumerate(to_be_found):
        if symbol != ' ':
            encoded.insert(i, vocabulary[unknown_token])
    return encoded


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None
    with open(vocab_path, encoding='utf-8') as file_content:
        vocabulary = json.load(file_content)
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
    if (not isinstance(original_text, str)
            or not isinstance(vocabulary, dict | None)
            or not isinstance(start_of_word_token, str | None)
            or not isinstance(end_of_word_token, str | None)
            or not isinstance(unknown_token, str)):
        return None
    words = original_text.split()
    encoded_text = []
    for word in words:
        unigrams = prepare_word(word, start_of_word_token, end_of_word_token)
        if not unigrams:
            return None
        tokens = tokenize_word(unigrams, vocabulary, end_of_word_token, unknown_token)
        if not tokens:
            return None
        encoded_text.extend(tokens)
    return encoded_text


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extracts n-grams from the given sequence
    :param text: original text
    :param order: required number of elements in a single n-gram
    :return: sequence of n-grams
    """
    if (not isinstance(text, str)
       or not isinstance(order, int)):
        return None
    chars = list(text)
    ngrams = [tuple(chars[i:i + order]) for i in range(len(chars) - order + 1)]
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
    if (not isinstance(actual, list)
       or not isinstance(reference, list)):
        return None
    if not actual:
        return 0
    return len(set(actual).intersection(reference))/len(set(reference))


def geo_mean(precisions: list[float], max_order: int) -> float | None:
    """
    Computes geometric mean of sequence of values
    :param precisions: sequence of Precision values
    :param max_order: maximum length of n-gram considered
    :return: value of geometric mean of Precision metric
    """
    if (not isinstance(precisions, list)
        or not isinstance(max_order, int)):
        return None
    if not all(precisions):
        return 0
    return math.exp(1 / max_order * sum(math.log(x) for x in precisions[:max_order]))


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compares two sequences by virtue of BLEU metric
    :param actual: predicted sequence
    :param reference: expected sequence
    :param max_order: max length of n-gram to consider for comparison
    :return: value of BLEU metric
    """
    if (not isinstance(actual, str | None)
       or not isinstance(reference, str)
       or not isinstance(max_order, int)):
        return None
    precisions = []
    for i in range(max_order):
        act_ngrams = collect_ngrams(actual, i + 1)
        ref_ngrams = collect_ngrams(reference, i + 1)
        if not act_ngrams or not ref_ngrams:
            return None
        metric = calculate_precision(act_ngrams, ref_ngrams)
        if not metric:
            return None
        precisions.append(metric)
    bleu = geo_mean(precisions, max_order)
    if not bleu:
        return None
    return bleu * 100
