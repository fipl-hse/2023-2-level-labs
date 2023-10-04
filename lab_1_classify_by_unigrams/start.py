"""
Language detection starter
"""


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    result = None
    assert result, "Detection result is None"

    def tokenize(text: str) -> list[str] | None:
        """
        Splits a text into tokens, converts the tokens into lowercase,
        removes punctuation, digits and other symbols
        :param text: a text
        :return: a list of lower-cased tokens without punctuation
        """
        if not isinstance(text, str):
            return None
        text = text.lower()
        wronglist = (list(text))
        goodlist = []
        for item in wronglist:
            if item.isalpha():
                goodlist.append(item)
        return (goodlist)

    tokenize("texugyTTRETHFDFG546//t")

    def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
        """
        Calculates frequencies of given tokens
        :param tokens: a list of tokens
        :return: a dictionary with frequencies
        """
        if not isinstance(tokens, list):
            return None
        numberallitems = len(tokens)
        freq_dict = {}
        for item in tokens:
            numberi = tokens.count(item)
            freqi = numberi / numberallitems
            freq_dict[item] = freqi
        return (freq_dict)

    calculate_frequencies(['t', 'e', 'x', 'u', 'g', 'y', 't', 't', 'r', 'e', 't', 'h', 'f', 'd', 'f', 'g', 't'])

    def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
        """
        Creates a language profile
        :param language: a language
        :param text: a text
        :return: a dictionary with two keys – name, freq
        """
        if not isinstance(language, str):
            return None
        if not isinstance(text, str):
            return None
        goodlist = tokenize(text)
        freq_dict = calculate_frequencies(goodlist)
        profilel = {"name": language, "freq": {}}

        for key, value in freq_dict.items():
            profilel["freq"][(key)] = (value)
        return (profilel)

    create_language_profile("en", "yft45dytdTU")

    def calculate_mse(predicted: list, actual: list) -> float | None:
        """
        Calculates mean squared error between predicted and actual values
        :param predicted: a list of predicted values
        :param actual: a list of actual values
        :return: the score
        """
        if not isistance(predicted, list):
            return None
        if not isinstance(actual, list):
            return None
        if len(predicted) != len(actual):
            return None
        if len(actual) == 0:
            return None
        else:
            dlina = len(actual)
            for index in range(dlina):
                numeratory += actual[index]
                numeratorp += predicted[index]
                numerator += (numeratory - numeratorp) ** 2
        mse = (numerator / dlina)


if __name__ == "__main__":
    main()
