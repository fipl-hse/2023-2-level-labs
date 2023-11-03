"""
Lab 2
BPE and machine translation evaluation
"""


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
    if not isinstance(raw_word, str) or not isinstance(start_of_word and end_of_word, (str | None)):
        return None
    word = list(raw_word)
    if start_of_word is not None:
        word.insert(0, start_of_word)
    if end_of_word is not None:
        word.append(end_of_word)
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
    if not isinstance(text, str) \
            or not isinstance(start_of_word, (str | None)) \
            or not isinstance(end_of_word, str):
        return None
    splited = text.split()
    freq = {}
    for i, word in enumerate(splited):
        preproc_word = prepare_word(word, start_of_word, end_of_word)
        if preproc_word is None:
            return None
        if preproc_word not in freq:
            freq[preproc_word] = 0
        freq[preproc_word] += 1
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
    pair_freq = {}
    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            token_pair = (word[i], word[i + 1])
            if token_pair not in pair_freq:
                pair_freq[token_pair] = 0
            pair_freq[token_pair] += freq
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
    if not isinstance(word_frequencies, dict) or not isinstance(pair, tuple):
        return None
    str_pair = "".join(pair)
    new_dict = {}
    for word, freq in word_frequencies.items():
        new_word = list(word)
        pair_indexes = []
        for i in range(1, len(word)):
            if (word[i - 1], word[i]) == pair:
                new_word[i-1:i+1] = [str_pair]
        new_dict.update({tuple(new_word): freq})
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
    if not isinstance(word_frequencies, dict) or not isinstance(num_merges, int):
        return None
    merges_count = 0
    while merges_count < num_merges:
        pair_freq = count_tokens_pairs(word_frequencies)
        if pair_freq is None:
            return None
        if len(pair_freq) == 0:
            break
        sorted_pairs = sorted(pair_freq.items(),
                             key=lambda item: (-item[1], -len("".join(item[0])), "".join(item[0])))
        word_frequencies = merge_tokens(word_frequencies, sorted_pairs[0][0])
        if word_frequencies is None:
            return None
        merges_count += 1
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
    tokens_set = set()
    tokens_set.add(unknown_token)
    for word in word_frequencies.keys():
        for token in word:
            tokens_set.add(token)
            if len(token) > 1:
                for symbol in token:
                    tokens_set.add(symbol)
    sorted_tokens_set = sorted(tokens_set, key=lambda item: (-len(item), item))
    vocabulary = {}
    for i, value in enumerate(sorted_tokens_set):
        vocabulary.update({value: i})
    return vocabulary


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
            or not isinstance(end_of_word_token, (str | None)):
        return None
    decoded_list = []
    for code in encoded_text:
        for k, value in vocabulary.items():
            if code == value:
                decoded_list.append(k)
    decoded_text = "".join(decoded_list)
    if end_of_word_token is not None:
        final_text = decoded_text.replace(end_of_word_token, ' ')
        return final_text
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


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Reads and retrieves dictionary of type <token: identifier>
    :param vocab_path: path to the saved vocabulary
    :return: dictionary in the form of <token: identifier>
    """


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
