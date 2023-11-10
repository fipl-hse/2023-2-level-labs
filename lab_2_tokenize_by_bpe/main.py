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
    if (not isinstance(raw_word, str) or not (
            isinstance(start_of_word, str) or start_of_word is None) or not (
            isinstance(end_of_word, str) or end_of_word is None)):
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
    if not isinstance(end_of_word, str) or not isinstance(
            text, str) or not (isinstance(
        start_of_word, str) or start_of_word is None):
        return None
    frequency = {}
    words_list = text.split()
    for word in words_list:
        prepared_word = prepare_word(word, start_of_word, end_of_word)
        if not prepared_word:
            return None
        frequency[prepared_word] = words_list.count(word)
    return frequency


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
    pairs_dict = {}
    for key, frequency in word_frequencies.items():
        for ind in range(len(key) - 1):
            pair = (key[ind], key[ind + 1])
            if pair not in pairs_dict:
                pairs_dict[pair] = 0
            pairs_dict[pair] += frequency
    return pairs_dict


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
    updated_dict = {}
    pair_tokens = ''.join(pair)
    for key, frequency in word_frequencies.items():
        if pair_tokens in ''.join(key):
            updated_key = list(key)
            for ind, token in enumerate(key[:-1]):
                if (token, key[ind + 1]) == pair:
                    updated_key[ind] = pair_tokens
                    del updated_key[ind + 1]
            updated_dict[tuple(updated_key)] = frequency
        else:
            updated_dict[key] = frequency
    return updated_dict


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
    while num_merges != 0:
        token_pairs = count_tokens_pairs(word_frequencies)
        if not token_pairs:
            return None
        num_merges = min(num_merges, len(token_pairs))
        max_num = max(token_pairs.values())
        new_pairs = [key for key, value in token_pairs.items() if value == max_num]
        longest = max(len(''.join(token)) for token in new_pairs)
        longest_token = [token for token in new_pairs if len(''.join(token)) == longest]
        final_token = sorted(longest_token)[0]
        word_frequencies = merge_tokens(word_frequencies, final_token)
        if not word_frequencies:
            return None
        num_merges -= 1
    return word_frequencies

def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establishes correspondence between set_tokens and its integer identifier
    :param word_frequencies: dictionary in the form of <preprocessed word: number of occurrences>
    :param unknown_token: a token to signify an unknown token
    :return: dictionary in the form of <token: identifier>
    """
    if not isinstance(word_frequencies, dict) or not isinstance(unknown_token, str):
        return None
    set_tokens = set()
    set_tokens.add(unknown_token)
    for keys in word_frequencies.keys():
        for token in keys:
            set_tokens.add(token)
            for symbol in token:
                set_tokens.add(symbol)
    new_tokens = sorted(set_tokens)
    new_tokens.sort(key=len, reverse=True)
    ident_tokens = {}
    for identifier, token in enumerate(new_tokens):
        ident_tokens[token] = identifier
    return ident_tokens

def decode(
    encoded_text: list[int] | None, vocabulary: dict[str, int] | None, end_of_word_token: str | None
) -> str | None:
    """
    Translates encoded sequence into decoded_text one
    :param encoded_text: a sequence of token identifiers
    :param vocabulary: dictionary in the form of <token: identifier>
    :param end_of_word_token: an end-of-word token
    :return: decoded_text sequence
    """
    if not isinstance(encoded_text, list) or not isinstance(
            vocabulary, dict) or not (isinstance(
        end_of_word_token, str) or end_of_word_token is None):
        return None
    decoded_text = ''
    for item in encoded_text:
        for token, identifier in vocabulary.items():
            if item == identifier:
                decoded_text += token
    if end_of_word_token:
        decoded_text = decoded_text.replace(end_of_word_token, ' ')

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
