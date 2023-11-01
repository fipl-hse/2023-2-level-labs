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
        and (isinstance(start_of_word, str) or start_of_word is None)
        and (isinstance(end_of_word, str) or end_of_word is None)
    ):
        return None


    word_tokens = list(raw_word)
    if start_of_word:
        word_tokens.insert(0, start_of_word)
    if end_of_word:
        word_tokens.append(end_of_word)

    return tuple(word_tokens)


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
        and isinstance(start_of_word, (str, type(None)))
        and isinstance(end_of_word, str)
    ):
        return None

    frequencies = {}
    words = text.split()

    for word in words:
        prepared_word = prepare_word(word, start_of_word, end_of_word)
        if prepared_word is None:
            return None

        frequencies[prepared_word] = words.count(word)

    return frequencies


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

    pair_frequency_dict = {}

    for word_tokens in word_frequencies:
        for i in range(len(word_tokens) - 1):
            token = word_tokens[i]
            next_token = word_tokens[i + 1]
            pair = (token, next_token)
            pair_frequency_dict[pair] = \
                pair_frequency_dict.get(pair, 0) + word_frequencies[word_tokens]

    return pair_frequency_dict


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
        isinstance(word_frequencies, dict)
        and isinstance(pair, tuple)
    ):
        return None

    merged_tokens_dict = {}

    for word in word_frequencies.keys():
        if ''.join(pair) in ''.join(word):
            list_word = list(word)
            for i in range(len(word) - 1):
                new_key = (word[i], word[i + 1])
                if new_key == pair:
                    list_word[i + 1] = ''.join(pair)
                    list_word.pop(i)
            merged_tokens_dict[tuple(list_word)] = word_frequencies[word]
        else:
            merged_tokens_dict[word] = word_frequencies[word]

    return merged_tokens_dict


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Creates required number of new tokens by merging existing ones
    :param word_frequencies: dictionary of a kind <preprocessed word: number of occurrences>
    :param num_merges: required number of new tokens
    :return: dictionary in the form of <preprocessed word: number of occurrences>
    """
    if not (isinstance(word_frequencies, dict) and isinstance(num_merges, int)
    ):
        return None

    while num_merges > 0:
        pair_frequency_dict = count_tokens_pairs(word_frequencies)
        if not pair_frequency_dict:
            return None

        if num_merges > len(pair_frequency_dict):
            num_merges = len(pair_frequency_dict)

        for i in range(num_merges):
            max_occurrence = max(pair_frequency_dict.values())
            max_occurrence_tokens = []
            for pair in pair_frequency_dict:
                if pair_frequency_dict[pair] == max_occurrence:
                    max_occurrence_tokens.append(pair)

            max_occurrence_tokens = sorted(max_occurrence_tokens,
                                           key=lambda x: (-len(''.join(x)), x))
            best_pair = max_occurrence_tokens[0]
            word_frequencies = merge_tokens(word_frequencies, best_pair)
            if not word_frequencies:
                return None

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
    if not (
        isinstance(word_frequencies, dict)
        and isinstance(unknown_token, str)
    ):
        return None

    vocabulary = {}
    tokens = set()
    for tuples in word_frequencies.keys():
        for token in tuples:
            tokens.add(token)
            for symbol in token:
                tokens.add(symbol)
    tokens.add(unknown_token)
    len_sorted = sorted(tokens, key=lambda x: (-len(x), x))
    for number, token in enumerate(len_sorted):
        vocabulary[token] = number
    return vocabulary


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
        isinstance(encoded_text, list)
        and all(isinstance(number, int) for number in encoded_text)
        and isinstance(vocabulary, dict)
        and (isinstance(end_of_word_token, str) or end_of_word_token is None)
    ):
        return None

    vocabulary_inverted = {id: token for token, id in vocabulary.items()}
    decoded_text = str()
    for number in encoded_text:
        token = vocabulary_inverted[number]
        if end_of_word_token and end_of_word_token in token:
            token = token.replace(end_of_word_token, ' ')
        decoded_text += token

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
