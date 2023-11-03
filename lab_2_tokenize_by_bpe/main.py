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
    if not (
        isinstance(raw_word, str) and
        (isinstance(start_of_word, str) or
         start_of_word is None) and
        (isinstance(end_of_word, str) or
         end_of_word is None)
    ):
        return None
    tokens = list(raw_word)
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
    if not (
        isinstance(text, str) and
        (isinstance(start_of_word, str) or
         start_of_word is None) and
        isinstance(end_of_word, str)
    ):
        return None
    list_of_tokens = text.split(' ')
    word_frequencies = {}
    for word in list_of_tokens:
        key = prepare_word(word, start_of_word, end_of_word)
        if not key:
            return None
        word_frequencies[key] = list_of_tokens.count(word)
    return word_frequencies


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
    pair_frequencies = {}
    for word in word_frequencies:
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            if pair not in pair_frequencies:
                pair_frequencies[pair] = 0
            pair_frequencies[pair] += word_frequencies[word]
    return pair_frequencies


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
        isinstance(word_frequencies, dict) and
        isinstance(pair, tuple)
    ):
        return None
    merged_pair = pair[0] + pair[1]
    new_word_freq = {}
    for word, freq in word_frequencies.items():
        new_word = list(word)
        for index in range(len(word) - 1):
            if (word[index], word[index + 1]) == pair:
                new_word[index] = merged_pair
                new_word[index + 1] = ''
            if '' in new_word:
                new_word.remove("")
        new_word_freq[tuple(new_word)] = freq
    return new_word_freq


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
        isinstance(word_frequencies, dict) and
        isinstance(num_merges, int)
    ):
        return None
    tokens_pairs = count_tokens_pairs(word_frequencies)
    if not tokens_pairs:
        return None
    for iteration in range(min(len(tokens_pairs), num_merges)):
        max_freq = max(tokens_pairs.values())
        max_freq_tokens = []
        for pair in tokens_pairs:
            if tokens_pairs[pair] == max_freq:
                max_freq_tokens.append(pair)
        max_len = max(len(pair[0] + pair[1]) for pair in max_freq_tokens)
        max_len_tokens = []
        for pair in max_freq_tokens:
            if max_len == len(pair[0] + pair[1]):
                max_len_tokens.append(pair)
        word_frequencies = merge_tokens(word_frequencies, sorted(max_len_tokens)[0])
        if not word_frequencies:
            return None
        tokens_pairs = count_tokens_pairs(word_frequencies)
        if not tokens_pairs:
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
        isinstance(word_frequencies, dict) and
        isinstance(unknown_token, str)
    ):
        return None
    tokens = set()
    for token in word_frequencies:
        for symbol in token:
            tokens.update(token, symbol)
    tokens.add(unknown_token)
    sorted_tokens = sorted(sorted(tokens), key=len, reverse=True)
    dict_ident = {}
    for index, token in enumerate(sorted_tokens):
        dict_ident[str(token)] = index
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
    if not (
        isinstance(encoded_text, list) and
        isinstance(vocabulary, dict) and
        (isinstance(end_of_word_token, str) or
         end_of_word_token is None)
    ):
        return None
    reverse_vocabulary = {}
    for token, ident in vocabulary.items():
        if end_of_word_token:
            reverse_vocabulary[ident] = token.replace(end_of_word_token, ' ')
        else:
            reverse_vocabulary[ident] = token
    decoded_seq = ''
    for num in encoded_text:
        decoded_seq += reverse_vocabulary[num]
    return decoded_seq


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
