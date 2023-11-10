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
            isinstance(raw_word, str)
            and isinstance(start_of_word, str | None)
            and isinstance(end_of_word, str | None)
    ):
        return None

    list_of_tokens = []

    if start_of_word is not None:
        list_of_tokens.append(start_of_word)

    list_of_tokens.extend(list(raw_word))

    if end_of_word is not None:
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
    if not (
            isinstance(text, str)
            and isinstance(start_of_word, str | None)
            and isinstance(end_of_word, str)
    ):
        return None

    words_list = text.split()
    word_frequencies = {}

    for word in words_list:
        processed_word = prepare_word(word, start_of_word, end_of_word)

        if processed_word is None:
            return None
        elif processed_word in word_frequencies:
            word_frequencies[processed_word] += 1
        else:
            word_frequencies[processed_word] = 1

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

    pair_of_tokens_frequencies = {}

    for word, frequency in word_frequencies.items():

        for letter_index in range(len(word) - 1):

            if word[letter_index] == '</s>':
                continue

            pair = (word[letter_index], word[letter_index + 1])

            if pair in pair_of_tokens_frequencies:
                pair_of_tokens_frequencies[pair] += frequency
            else:
                pair_of_tokens_frequencies[pair] = frequency

    return pair_of_tokens_frequencies


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

    updated_word_frequencies = {}

    for word in word_frequencies:
        word_in_list_type = list(word)

        for index in range(len(word_in_list_type) - 1):
            if (word[index], word[index + 1]) == pair:
                word_in_list_type[index + 1] = pair[0] + pair[1]
                word_in_list_type[index] = 'extra_symbol'

        if 'extra_symbol' in word_in_list_type:
            word_in_list_type.remove('extra_symbol')
            updated_word_frequencies.update({tuple(word_in_list_type): word_frequencies[word]})
        else:
            updated_word_frequencies.update({word: word_frequencies[word]})

    return updated_word_frequencies


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

    for _ in range(num_merges):

        tokens_pairs_frequencies = count_tokens_pairs(word_frequencies)
        if not tokens_pairs_frequencies:
            return None

        pair_to_merge_length = max(tokens_pairs_frequencies.values())

        longest_keys = []
        improved_longest_keys = []

        for key, value in tokens_pairs_frequencies.items():
            if value == pair_to_merge_length:
                longest_keys.append(key)

        for key in range(len(longest_keys)):
            for next_key in range(len(longest_keys)):
                if longest_keys[key] == longest_keys[next_key]:
                    continue
                extra_pair = max(longest_keys[key], longest_keys[next_key])
                if extra_pair == longest_keys[key]:
                    longest_keys[key] = tuple('❌' * pair_to_merge_length)
                elif extra_pair == longest_keys[next_key]:
                    longest_keys[next_key] = tuple('❌' * pair_to_merge_length)

        for i in longest_keys:
            if i == tuple('❌' * pair_to_merge_length):
                longest_keys.remove(i)

        word_frequencies = merge_tokens(word_frequencies, longest_keys[0])
        if not word_frequencies:
            return None

        tokens_pairs_frequencies = count_tokens_pairs(word_frequencies)
        if not tokens_pairs_frequencies:
            return None

    return word_frequencies


dict1 = collect_frequencies('про скупого сизого орла', None, '</s>')


num1 = 4

train(dict1, num1)

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

