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
    if (not isinstance(raw_word, str) or not 
        (isinstance(start_of_word, str) or start_of_word is None) or not 
        (isinstance(end_of_word, str) or end_of_word is None)):
        return None
    preprocessed_word = []
    if start_of_word != None:
        preprocessed_word.append(start_of_word)
    for token in raw_word:
        preprocessed_word.append(token)
    if end_of_word != None:
        preprocessed_word.append(end_of_word)
    return tuple(preprocessed_word)


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
    if (not isinstance(text, str) or not 
        (isinstance(start_of_word, str) or start_of_word is None) or not
        isinstance(end_of_word, str)):
        return None
    freq_dictionary = {}
    listed_text = text.split(" ")
    for word in listed_text:
        preprocessed_word = prepare_word(word, start_of_word, end_of_word)
        if preprocessed_word == None:
            return None
        if preprocessed_word not in freq_dictionary:
            freq_dictionary[preprocessed_word] = 0
        freq_dictionary[preprocessed_word] += 1
    return freq_dictionary


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
    listed_combinations = []
    combination_frequencies = {}
    for word in word_frequencies:
        for i, token in enumerate(word):
            if (i + 1) < len(word):
                listed_combinations.append((token, word[i + 1]))
    for combination in listed_combinations:
        if combination not in combination_frequencies:
            combination_frequencies[combination] = 0
        combination_frequencies[combination] += 1
    return combination_frequencies


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
    merged_frequencies = {}
    for word, freq in word_frequencies.items():
        if ''.join(pair) in ''.join(word):
            word = list(word)
            i = word.index(pair[0])
            word[i] = pair[0] + pair[1]
            word.pop(i + 1)
            word = tuple(word)
        merged_frequencies[word] = freq
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
    if (not (isinstance(word_frequencies, dict) or word_frequencies is None) or not
        isinstance(num_merges, int)):
        return None
    token_pairs = count_token_pairs(word_frequencies)
    while (num_merges > 0) and len(token_pairs) > 0:
        num_merges -= 1
        max_freq = 0
        max_pairs = []
        for freq in token_pairs.values():
            if freq > max_freq:
                max_freq = freq
        for pair in token_pairs:
            if token_pairs[pair] == max_freq:
                max_pairs.append(word)
        if len(max_pairs) > 1:
            long_pair = ""
            for pair in max_pairs:
                if len("".join(pair)) > len(long_pair):
                    long_pair = pair
            long_pairs = []
            for pair in max_pairs:
                if len(pair) == len(long_pair):
                    long_pairs.append(pair)
            if len(long_pairs) > 1:
                pair_to_merge = min(long_pairs)
                del token_pairs[pair_to_merge]
            else:
                pair_to_merge = long_pairs[0]
                del token_pairs[pair_to_merge]
        else:
            pair_to_merge = max_pairs[0]
            del token_pairs[pair_to_merge]
        word_frequencies = merge_tokens(wors_frequencies, pair_to_merge)
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
