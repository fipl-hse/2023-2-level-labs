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
    if (not isinstance(raw_word, str) or
        not (isinstance(start_of_word, str) or start_of_word is None) or
        not (isinstance(end_of_word, str) or end_of_word is None)
    ):
        return None

    return tuple(([start_of_word] if start_of_word else []) + list(raw_word) + ([end_of_word] if end_of_word else []))


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
        not (isinstance(start_of_word, str) or start_of_word is None) or
        not (isinstance(end_of_word, str))
    ):
        return None
    words = text.split()
    collection = {}
    for word in words:
        prepared = prepare_word(word, start_of_word, end_of_word)
        if prepared is None:
            return None
        if prepared not in collection:
            collection[prepared] = 0
        collection[prepared] += 1

    return collection


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
    pairs = {}
    for word in word_frequencies:
        for i in range(len(word)-1):
            pair = tuple([word[i], word[i+1]])
            if pair not in pairs:
                pairs[pair] = 0
            pairs[pair] += word_frequencies[word]

    return pairs


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not( isinstance(word_frequencies, dict) and
            isinstance(pair, tuple)
    ):
        return None

    new_word_frequencies = {}
    for word in word_frequencies:
        new_word = list(word)
        for i in range(len(word)-1):
            two_tokens = tuple([word[i], word[i+1]])
            if pair == two_tokens:
                new_word.pop(i+1)
                new_word[i] = pair[0] + pair[1]

        new_word = tuple(new_word)
        new_word_frequencies[new_word] = word_frequencies[word]

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
    if not (isinstance(word_frequencies, dict) and
            isinstance(num_merges, int)
    ):
        return None

    count_of_pairs = count_tokens_pairs(word_frequencies)
    if count_of_pairs is None:
        return None
    num_merges = min(num_merges, len(count_of_pairs))

    for i in range(num_merges):
        maximum = max(count_of_pairs.values())

        local_maximums = [list(key) for key in count_of_pairs if count_of_pairs[key] == maximum]
        max_length = max(len(''.join(tokens)) for tokens in local_maximums)

        longest_pairs = [p for p in local_maximums if len(''.join(p)) == max_length]
        pair = sorted(longest_pairs)[0]

        merged_freq = merge_tokens(word_frequencies, tuple(pair))
        if merged_freq is None:
            return None

        word_frequencies = merged_freq
        count_of_pairs = count_tokens_pairs(word_frequencies)
        if count_of_pairs is None:
            return None

    return merged_freq

def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establishes correspondence between tokens and its integer identifier
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param unknown_token: a token to signify an unknown token
    :return: dictionary in the form of <token: identifier>
    """


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
