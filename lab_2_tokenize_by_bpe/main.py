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
    if not isinstance(raw_word, str):
        return None
    if not (isinstance(start_of_word, str))\
            and start_of_word is not None:
        return None
    if not (isinstance(end_of_word, str))\
            and end_of_word is not None:
        return None
    tokens_list = list(raw_word)

    if start_of_word is not None:
        tokens_list.insert(0, start_of_word)

    if end_of_word is not None:
        tokens_list.append(end_of_word)
    return tuple(tokens_list)


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
    if not isinstance(text, str) or not isinstance(end_of_word, str)\
            or not (isinstance(start_of_word, str) or start_of_word is None):
        return None

    word_list = text.split()
    freq_dict = {}
    for word in word_list:
        freq = word_list.count(word)
        prep_word = prepare_word(word, start_of_word, end_of_word)
        if prep_word is None:
            return None
        if prep_word not in freq_dict:
            freq_dict.update({prep_word:freq})

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

    tok_pair_dict = {}
    for token in word_frequencies:
        for index in range(len(token)-1):
            tok_pair = (token[index], token[index+1])
            if tok_pair not in tok_pair_dict:
                tok_pair_dict[tok_pair] = 0
            tok_pair_dict[tok_pair] += word_frequencies[token]
    return tok_pair_dict
def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Updates word frequency dictionary by replacing a pair of token with a merged one
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param pair: a pair of tokens to be merged
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not isinstance(word_frequencies, dict)\
        or not isinstance(pair, tuple):
        return None

    merge_dict={}
    for word, freq in word_frequencies.items():
        tok = list(word)
        for index in range(len(word) - 1):
            if pair == (word[index], word[index+1]):
                tok[index] = str(pair[0]+pair[1])
                tok.pop(index+1)
        merge_dict[tuple(tok)] = freq
    return merge_dict

def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if (not isinstance(word_frequencies, dict)
            or not isinstance(num_merges, int)):
        return None
    while num_merges > 0:
        tok_pairs = count_tokens_pairs(word_frequencies)
        if not tok_pairs:
            return None
        num_merges = min(num_merges, len(tok_pairs))
        the_biggest = max(tok_pairs.values())
        com_pair = [key for key, value in tok_pairs.items()
                    if value == the_biggest]
        the_longest = max(len(str(pair)) for pair in com_pair)
        the_longest_pair =[pair for pair in com_pair
                           if the_longest == len(str(pair))]
        lovely_pairs = sorted(the_longest_pair)
        word_frequencies = merge_tokens(word_frequencies, lovely_pairs[0])
        if not word_frequencies:
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
    if not isinstance(word_frequencies, dict)\
        or not isinstance(unknown_token, str):
        return None
    tok_list = set()
    get_dict = {}
    for tuples in word_frequencies.keys():
        for word in tuples:
            tok_list.add(word)
            for token in word:
                tok_list.add(token)
    tok_list.add(unknown_token)
    lex_sort = sorted(tok_list)
    len_sort = sorted(lex_sort, key=len, reverse = True)
    for index, token in enumerate(len_sort):
        get_dict[token] = index
    return get_dict



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
    if not isinstance(encoded_text, list)\
            or not isinstance(vocabulary, dict)\
            or not (isinstance(end_of_word_token, str)\
            or end_of_word_token is None):
        return None
    decoder = ''
    for number in encoded_text:
        tok_l = [tok for tok in vocabulary if vocabulary[tok] == number]
        for tok in tok_l:
            decoder += tok
    if end_of_word_token:
        decoder = decoder.replace(end_of_word_token, ' ')

    return decoder


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
