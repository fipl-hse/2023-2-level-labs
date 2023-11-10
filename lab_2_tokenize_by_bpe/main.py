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
    if not isinstance(raw_word, str) or not isinstance(end_of_word, str | None) or not isinstance(start_of_word,
                                                                                                  str | None):
        return None

    raw_tokens = list(raw_word)
    if start_of_word != None:
        raw_tokens.insert(0, start_of_word)
    if end_of_word != None:
        raw_tokens.append(end_of_word)
    return tuple(raw_tokens)





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
    if not isinstance(text, str) or not isinstance(end_of_word, str):
        return None
    if not isinstance(start_of_word, str) and start_of_word != None:
        return None

    word_list = text.split()
    dictionary = {}
    for i in set(word_list):
        word_freq = word_list.count(i)
        word_prepared = prepare_word(i, start_of_word, end_of_word)
        if word_prepared == None:
            return None

        dictionary[word_prepared] = word_freq

    return dictionary


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

    result = {}

    for word, number in word_frequencies.items():
        for index, letter in enumerate(word[:-1]):
            pair = (word[index], word[index + 1])
            if pair not in result:
                result[pair] = 0
            result[pair] += number

    return result

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

    narrowed_dict = word_frequencies.copy()
    for key, value in word_frequencies.items():
        if pair[0] and pair[1] in key:
            changed_key = []
            for f_index in [i for i, c in enumerate(key) if c == pair[0]]:
                for s_index in [i for i, c in enumerate(key) if c == pair[1]]:
                    if f_index - s_index != -1:
                        continue
                    else:
                        pagination = lambda s, n: [s[i:i + n] for i in (range(0, len(s), n))]
                        N = 1
                        full_list = pagination("".join(pair), 100)
                        the_pair = [tuple(full_list[n:n + N]) for n in range(0, len(full_list), N)]
                        changed_key = tuple(key[:f_index]) + the_pair[0] + tuple(key[f_index + 2:])
                    narrowed_dict.pop(key)
                    narrowed_dict[tuple(changed_key)] = value

    return narrowed_dict


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(num_merges, int):
        return None
    if not isinstance(word_frequencies, dict) and not isinstance(num_merges, int):
        return None

    the_pairs = count_tokens_pairs(word_frequencies)
    if not the_pairs:
        return None
    num_merges = min(num_merges, len(the_pairs))
    for num_merged in range(num_merges):

        possible_pair = [k for k, value in the_pairs.items() if value == max(the_pairs.values())]
        longest_pairs = max(len(''.join(pair)) for pair in possible_pair)
        probable_pair = [pair for pair in possible_pair if longest_pairs == len(''.join(pair))]

        word_frequencies = merge_tokens(word_frequencies, sorted(probable_pair)[0])
        if not word_frequencies:
            return None
        the_pairs.pop(sorted(probable_pair)[0])
        the_pairs = count_tokens_pairs(word_frequencies)
        if not the_pairs:
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
