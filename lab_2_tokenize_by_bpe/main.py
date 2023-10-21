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
    if not (
        isinstance(raw_word, str)
        and (isinstance(start_of_word, str) or start_of_word is None)
        and (isinstance(end_of_word, str) or end_of_word is None)
    ):
        return None

    tokenized_word = list(raw_word)
    if start_of_word:
        tokenized_word.insert(0, start_of_word)
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
    if not (
        isinstance(text, str)
        and (isinstance(start_of_word, str) or start_of_word is None)
        and isinstance(end_of_word, str)
    ):
        return None

    frequency_dict = {}
    text_list = text.split()
    for word in text_list:
        prepared_word = prepare_word(word, start_of_word, end_of_word)
        if not prepared_word:
            return None

        frequency_dict[prepared_word] = text_list.count(word)

    return frequency_dict


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

    pair_frequency_dict = {}

    for word_tokens in word_frequencies:
        for i in range(len(word_tokens) - 1):
            pair_token = word_tokens[i:i + 2]

            if pair_token not in pair_frequency_dict:
                pair_frequency_dict[pair_token] = 0

            pair_frequency_dict[pair_token] += word_frequencies[word_tokens]

    return pair_frequency_dict


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not(
        isinstance(word_frequencies, dict)
        and isinstance(pair, tuple)
    ):
        return None

    merged_dict = {}
    second_token = pair[1]

    for word_tokens in word_frequencies:
        word_tokens_list = list(word_tokens)
        if second_token in word_tokens:
            pair_token_index = []
            for index in range(len(word_tokens) - 1):
                token_pair = word_tokens[index: index + 2]
                if token_pair == pair:
                    pair_token_index.append(index)

            for pair_index in reversed(pair_token_index):
                word_tokens_list[pair_index: pair_index + 2] = [''.join(pair)]

            merged_dict[tuple(word_tokens_list)] = word_frequencies[word_tokens]

        else:
            merged_dict[word_tokens] = word_frequencies[word_tokens]

    return merged_dict


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

    token_pairs_dict = count_tokens_pairs(word_frequencies)
    if not token_pairs_dict:
        return None

    if num_merges > len(token_pairs_dict):
        num_merges = len(token_pairs_dict)

    for i in range(num_merges):
        max_freq = max(token_pairs_dict.values())
        max_freq_tokens = []

        for pair in token_pairs_dict:
            if token_pairs_dict[pair] == max_freq:
                max_freq_tokens.append(pair)

        max_freq_tokens = sorted(max_freq_tokens, key=lambda x: (-len(''.join(x)), x))
        token_pair = max_freq_tokens[0]

        word_frequencies = merge_tokens(word_frequencies, token_pair)

        if not word_frequencies:
            return None

        token_pairs_dict = count_tokens_pairs(word_frequencies)
        if not token_pairs_dict:
            return None

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
    for word_tokens in word_frequencies:
        for token in word_tokens:
            all_tokens.append(token)
            prepared_word = prepare_word(token, None, None)
            all_tokens += prepared_word
    all_tokens_unique = set(all_tokens)
    all_tokens_sorted = sorted(all_tokens_unique, key=lambda x: (-len(x), x))

    tokens_id_dict = {}
    for i, token in enumerate(all_tokens_sorted):
        tokens_id_dict[token] = i

    return tokens_id_dict


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
    if not (
        isinstance(encoded_text, list) and all(isinstance(numb, int) for numb in encoded_text)
        and isinstance(vocabulary, dict)
        and (isinstance(end_of_word_token, str) or end_of_word_token is None)
    ):
        return None

    vocab_inverted = {ident: token for token, ident in vocabulary.items()}
    decoded_text = ''
    for number in encoded_text:
        token = vocab_inverted[number]
        if end_of_word_token and end_of_word_token in token:
            token = token.replace(end_of_word_token, ' ')
        decoded_text += token

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
    if not (
        isinstance(word, tuple) and all(isinstance(token, str) for token in word)
        and isinstance(vocabulary, dict) and all(isinstance(token, str) for token in vocabulary)
        and all(isinstance(ident, int) for ident in vocabulary.values())
        and (isinstance(end_of_word, str) or end_of_word is None)
        and isinstance(unknown_token, str)
    ):
        return None

    word_str = ''.join(word)
    tokens_list = list(vocabulary.keys())
    tokens_list = sorted(tokens_list, key=lambda x: (-len(x), x))
    encoded_word = []

    for token in tokens_list:
        if token in word_str:
            encoded_word.append(vocabulary[token])
            word_str = word_str.replace(token, '')
    if len(word_str) > 0:
        encoded_word.append(vocabulary['<unk>'])

    return encoded_word


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, 'r',  encoding="utf-8") as vocab_file:
        large_vocab = json.load(vocab_file)

    if not isinstance(large_vocab, dict):
        return None

    return large_vocab


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
    if not (
        isinstance(original_text, str)
        and isinstance(vocabulary, dict)
        and (isinstance(start_of_word_token, str) or start_of_word_token is None)
        and (isinstance(end_of_word_token, str) or end_of_word_token is None)
        and isinstance(unknown_token, str)
    ):
        return None

    text_list = original_text.split()
    prepared_words = []
    for word in text_list:
        prepared_word = prepare_word(word, start_of_word_token, end_of_word_token)
        if not prepared_word:
            return None

        prepared_words.append(prepared_word)

    encoded_text_large = []
    for word_tokens in prepared_words:
        tokenized_word = tokenize_word(word_tokens, vocabulary, end_of_word_token, unknown_token)
        if not tokenized_word:
            return None

        encoded_text_large.extend(tokenized_word)

    return encoded_text_large


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
