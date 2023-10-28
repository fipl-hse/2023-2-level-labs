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
    if not ((isinstance(raw_word, str)
            and (isinstance(start_of_word, str) or start_of_word is None)
            and (isinstance(end_of_word, str) or end_of_word is None))):
        return None
    prepared_word = list(raw_word)
    if start_of_word:
        prepared_word.insert(0, start_of_word)
    if end_of_word:
        prepared_word.append(end_of_word)
    return tuple(prepared_word)


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
    if not (isinstance(text, str)
            and (isinstance(start_of_word, str) or start_of_word is None)
            and isinstance(end_of_word, str)):
        return None
    freq_dict = {}
    text_split = text.split()
    prepared_words = [prepare_word(word, start_of_word, end_of_word) for word in text_split]

    for word in set(prepared_words):
        if not word:
            return None
        freq_dict[word] = prepared_words.count(word)
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
    pair_freq = {}
    for key in word_frequencies:
        for token_index in range(len(key) - 1):
            tokens = (key[token_index], key[token_index + 1])
            if tokens not in pair_freq:
                pair_freq[tokens] = 0
            pair_freq[tokens] += word_frequencies[key]
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
    if not (isinstance(word_frequencies, dict)
            and isinstance(pair, tuple)):
        return None
    new_freq_dict = {}
    for word in word_frequencies.keys():
        if ''.join(pair) in ''.join(word):
            list_word = list(word)
            for ind in range(len(word) - 1):
                new_key = (word[ind], word[ind + 1])
                if new_key == pair:
                    list_word[ind + 1] = ''.join(pair)
                    list_word.pop(ind)
            new_freq_dict[tuple(list_word)] = word_frequencies[word]
        else:
            new_freq_dict[word] = word_frequencies[word]
    return new_freq_dict


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not (isinstance(word_frequencies, dict)
            and isinstance(num_merges, int)):
        return None
    while num_merges > 0:
        paired_words = count_tokens_pairs(word_frequencies)
        if not paired_words:
            return None
        num_merges = min(num_merges, len(paired_words))
        max_occur = max(paired_words.values())
        good_pairs = [pair for pair, freq in paired_words.items() if freq == max_occur]
        max_len = max(len(str(pair)) for pair in good_pairs)
        longest_pairs = [pair for pair in good_pairs if len(str(pair)) == max_len]
        best_pair = sorted(longest_pairs)
        word_frequencies = merge_tokens(word_frequencies, best_pair[0])
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
    if not (isinstance(word_frequencies, dict)
            and isinstance(unknown_token, str)):
        return None
    vocab_dict = {}
    not_good_list = set()
    for tuples in word_frequencies.keys():
        for tkn in tuples:
            not_good_list.add(tkn)
            for symb in tkn:
                not_good_list.add(symb)
    not_good_list.add(unknown_token)
    len_sorted = sorted(not_good_list, key=lambda x: (-len(x), x))
    for num, token in enumerate(len_sorted):
        vocab_dict[token] = num
    return vocab_dict


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
    if not (isinstance(encoded_text, list) and all(isinstance(numb, int) for numb in encoded_text)
            and isinstance(vocabulary, dict)
            and (isinstance(end_of_word_token, str) or end_of_word_token is None)):
        return None
    text = ''
    inv_d = {value: key for key, value in vocabulary.items()}
    for num in encoded_text:
        if end_of_word_token and end_of_word_token in inv_d[num]:
            text += inv_d[num].replace(end_of_word_token, ' ')
        elif num in vocabulary.values():
            text += inv_d[num]
    return text


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
