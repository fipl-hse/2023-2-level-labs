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
    if (
        not isinstance(raw_word, str)
        or not (start_of_word is None or isinstance(start_of_word, str))
        or not (end_of_word is None or isinstance(end_of_word, str))
    ):
        return None
    tokens = []
    if start_of_word:
        tokens.insert(0, start_of_word)
    tokens.extend(list(raw_word))
    if end_of_word:
        tokens.append(end_of_word)
    preprocessed_word = tuple(tokens)

    return preprocessed_word

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
    if (not isinstance(text, str)
            or (start_of_word is not None and not isinstance(start_of_word, str))
            or not isinstance(end_of_word, str)
    ):
        return None
    freq_dict = {}
    text = text.split()
    for preprocessed_word in text:
        preprocessed_word = prepare_word(preprocessed_word, start_of_word, end_of_word)
        if preprocessed_word is None:
            return None
        if preprocessed_word not in freq_dict:
            freq_dict.update({preprocessed_word: text.count(preprocessed_word)})
        freq_dict[preprocessed_word] += 1

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
    pair_frequencies = {}
    for word, freq in word_frequencies.items():
        for i in range(len(word)-1):
            tokens_pair = word[i], word[i+1]
            tokens_pair = tuple(tokens_pair)
            if tokens_pair not in pair_frequencies:
                pair_frequencies[tokens_pair] = 0
            pair_frequencies[tokens_pair] += freq

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
    if (not isinstance(word_frequencies, dict)
        or not isinstance(pair, tuple)
    ):
        return None
    new_word_freq = {}
    for word in word_frequencies:
        new_word = list(word)
        for i in range(len(word)-1):
            word_pair = tuple([word[i], word[i+1]])
            if pair == word_pair:
                new_word[i] = ''.join([word[i], word[i + 1]])
                new_word.pop(i + 1)
        new_word_freq[tuple(new_word)] = word_frequencies[word]

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
    if not (isinstance(word_frequencies, dict)
            and isinstance(num_merges, int)
    ):
        return None

    count_pairs = count_tokens_pairs(word_frequencies)
    if count_pairs is None:
        return None
    if num_merges > len(count_pairs):
        num_merges = len(count_pairs)

    for i in range(num_merges):
        most_freq = max(count_pairs.values())
        pair_list = [key for key, value in count_pairs.items() if value == most_freq]
        joined_pair = [''.join(pair) for pair in pair_list]

        maximum = max(len(pair) for pair in joined_pair)
        max_len_pairs = [pair for pair in pair_list if len(''.join(pair)) == maximum]

        pair = sorted(max_len_pairs)[0]

        word_frequencies = merge_tokens(word_frequencies, pair)
        if word_frequencies is None:
            return None
        count_pairs = count_tokens_pairs(word_frequencies)
        if count_pairs is None:
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
    if not (isinstance(word_frequencies, dict)
            and isinstance(unknown_token, str)
    ):
        return None
    identifiers = {}
    tokens = set()
    for word in word_frequencies:
        for token in word:
            tokens.update(word)
            tokens.update(token)

    tokens.add(unknown_token)
    lex_sorted_tokens = sorted(tokens)
    sorted_tokens = sorted(lex_sorted_tokens, key=len, reverse=True)

    for index, element in enumerate(sorted_tokens):
        identifiers[element] = index

    return identifiers

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
    if not (isinstance(encoded_text, list)
            and isinstance(vocabulary, dict)
            and (end_of_word_token is None or isinstance(end_of_word_token, str))
    ):
        return None
    decoded_tokens = []
    for identifier in encoded_text:
        for key, value in vocabulary.items():
            if value == identifier:
                decoded_tokens.append(key)
    decoded_text = ''.join(decoded_tokens)

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
