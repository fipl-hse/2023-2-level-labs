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
        and (start_of_word is None or isinstance(start_of_word, str))
        and (end_of_word is None or isinstance(end_of_word, str))
    ):
        return None
    total_result = []
    if start_of_word:
        total_result.append(start_of_word)
    for token in raw_word:
        total_result.append(token)
    if end_of_word:
        total_result.append(end_of_word)
    return tuple(total_result)


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
    if not (isinstance(text, str) and isinstance(end_of_word, str)) or not (
            isinstance(start_of_word, str) or start_of_word is None
    ):
        return None
    freq_dict = {}
    words = text.split()
    for word in words:
        preprocessed_word = prepare_word(word, start_of_word, end_of_word)
        if preprocessed_word is None:
            return None
        if preprocessed_word not in freq_dict:
            freq_dict[preprocessed_word] = 0
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
    pairs_of_tokens = {}
    for word, freq in word_frequencies.items():
        for ind in range(len(word) - 1):
            token_pair = (word[ind], word[ind + 1])
            if token_pair not in pairs_of_tokens:
                pairs_of_tokens[token_pair] = 0
            pairs_of_tokens[token_pair] += freq
    return pairs_of_tokens


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
    merged_dict = {}
    for word, freq in word_frequencies.items():
        new_word = [pair[0] + pair[1] if (word[i], word[i + 1]) == pair else word[i] for i in range(len(word) - 1)]
        new_word.append(word[-1])
        merged_dict[tuple(new_word)] = freq
    return merged_dict


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
        pairs_of_tokens = count_tokens_pairs(word_frequencies)
        if not pairs_of_tokens:
            return None
        if num_merges > len(pairs_of_tokens):
            num_merges = len(pairs_of_tokens)
        max_prevalence = max(pairs_of_tokens.values())
        max_prev_pairs = [pair for pair in pairs_of_tokens if pairs_of_tokens[pair] == max_prevalence]
        max_length = max([len(str(pair)) for pair in max_prev_pairs])
        max_length_pairs = [pair for pair in max_prev_pairs if len(str(pair)) == max_length]
        result = sorted(max_length_pairs)
        word_frequencies = merge_tokens(word_frequencies, result[0])
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
    if not isinstance(word_frequencies, dict) or not isinstance(unknown_token, str):
        return None
    ident_dict = {}
    tokens = set()
    for word_tuples in word_frequencies.keys():
        for word in word_tuples:
            tokens.add(word)
            for token in word:
                tokens.add(token)
    tokens.add(unknown_token)
    sorted_tokens = sorted(list(tokens), key=lambda uni_token: (-len(uni_token), uni_token))
    for ind, token in enumerate(sorted_tokens):
        ident_dict[token] = ind
    return ident_dict


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
            and not all(isinstance(char, int) for char in encoded_text)
            and not isinstance(vocabulary, dict)
            and not (isinstance(end_of_word_token, str) or end_of_word_token is None)):
        return None
    decoded_list = []
    for num in encoded_text:
        for key, value in vocabulary.items():
            if num == value:
                decoded_list.append(key)
    decoded_text = ''.join(decoded_list)
    if end_of_word_token:
        decoded_text = decoded_text.replace(end_of_word_token, ' ')
    return decoded_text.strip()


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
