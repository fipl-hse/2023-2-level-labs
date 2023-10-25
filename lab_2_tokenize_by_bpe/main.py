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
    if not isinstance(raw_word, str) or\
            not (isinstance(start_of_word, str) or start_of_word is None) or\
            not (isinstance(end_of_word, str) or end_of_word is None):
        return None
    token_list = list(raw_word)
    if start_of_word is not None:
        token_list.insert(0, start_of_word)
    if end_of_word is not None:
        token_list.append(end_of_word)
    return tuple(token_list)



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
    if not isinstance(text, str) or\
            not isinstance(start_of_word, str | None) or\
            not isinstance(end_of_word, str):
        return None
    tokens_dictionary = {}
    for token in text.split():
        split_word = prepare_word(token, start_of_word, end_of_word)
        if split_word is None:
            return None
        if split_word not in tokens_dictionary:
            tokens_dictionary[split_word] = text.split().count(token)
    return tokens_dictionary


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
    pairs_dictionary = {}
    for token in word_frequencies:
        for index in range(len(token) - 1):
            token_pair = (token[index], token[index + 1])
            if token_pair not in pairs_dictionary:
                pairs_dictionary[token_pair] = 0
            pairs_dictionary[token_pair] += word_frequencies[token]
    return pairs_dictionary

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
    merge_token_dictionary = {}
    for token in word_frequencies:
        token_list = list(token)
        if pair[0] in token and pair[1] in token:
            for index in range(len(token_list) - 1):
                if (token[index], token[index + 1]) == pair:
                    token_list[index: index + 2] = [''.join(pair)]
            merge_token_dictionary[tuple(token_list)] = word_frequencies[token]
        else:
            merge_token_dictionary[token] = word_frequencies[token]
    return merge_token_dictionary

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
        num_of_pairs = count_tokens_pairs(word_frequencies)
        if num_of_pairs is None:
            return None
        if num_merges > len(num_of_pairs):
            num_merges = len(num_of_pairs)
        max_freq = max(num_of_pairs.values())
        pairs_list = []
        for pair, frequency in num_of_pairs.items():
            if frequency == max_freq:
                pairs_list.append(pair)
        word_frequencies = merge_tokens(word_frequencies, (sorted(pairs_list,key = len ))[0])
        if word_frequencies is None:
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
    if not isinstance(word_frequencies, dict) or\
            not isinstance(unknown_token, str):
        return None
    list_of_tokens = set()
    get_vocabulary_dict = {}
    for tuples in word_frequencies.keys():
        for word in tuples:
            list_of_tokens.add(word)
            for token in word:
                list_of_tokens.add(token)
    list_of_tokens.add(unknown_token)
    right_order = sorted(list_of_tokens, key=lambda item: (-len(item), item))
    for index, token in enumerate(right_order):
        get_vocabulary_dict[token] = index
    return get_vocabulary_dict

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
    if not (isinstance(encoded_text, list) or encoded_text is None) or\
            not (isinstance(vocabulary, dict) or vocabulary is None) or\
            not (isinstance(end_of_word_token, str) or end_of_word_token is None):
        return None



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
