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
        wronglist = (list(text.lower()))
        goodlist = []
        for item in wronglist:
            if item.isalpha():
                goodlist.append(item)
        return (goodlist)

    def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
        """
        Calculates frequencies of given tokens
        :param tokens: a list of tokens
        :return: a dictionary with frequencies
        """
        if not isinstance(tokens, list):
            return None
        else:
            for token in tokens:
                if not isinstance(token, str):
                    return None
                else:
                    numberallitems = len(tokens)
                    freq_dict = {}
                    for item in tokens:
                        numberi = tokens.count(item)
                        freqi = numberi / numberallitems
                        freq_dict[item] = freqi
        return freq_dict

    def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
        """
        Creates a language profile
        :param language: a language
        :param text: a text
        :return: a dictionary with two keys – name, freq
        """
        if not isinstance(language, str):
            return None
        elif not isinstance(text, str):
            return None
        else:
            goodlist = tokenize(text)
            freq_dict = calculate_frequencies(goodlist)
            profilel = {"name": language, "freq": {}}
            for key, value in freq_dict.items():
                profilel["freq"][key] = (value)
        return (profilel)

    def calculate_mse(predicted: list, actual: list) -> float | None:
        """
        Calculates mean squared error between predicted and actual values
        :param predicted: a list of predicted values
        :param actual: a list of actual values
        :return: the score
        """
        if not isinstance(predicted, list) or not isinstance(actual, list) or len(predicted) != len(actual):
            return None
        else:
            numerator = 0
            dlina = len(actual)
            for index in range(dlina):
                numerator += (actual[index] - predicted[index]) ** 2
            mse = (numerator / dlina)
            return mse

    def compare_profiles(
            unknown_profile: dict[str, str | dict[str, float]],
            profile_to_compare: dict[str, str | dict[str, float]]) -> float | None:
        """
        Compares profiles and calculates the distance using symbols
        :param unknown_profile: a dictionary of an unknown profile
        :param profile_to_compare: a dictionary of a profile to compare the unknown profile to
        :return: the distance between the profiles
        """
        if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
            return None
        if "name" not in unknown_profile or "freq" not in unknown_profile:
            return None
        if "name" not in profile_to_compare or "freq" not in profile_to_compare:
            return None
        else:
            united_keys = []
            unknown_profile_frequency = []
            profile_to_compare_frequency = []
            for key in unknown_profile.get("freq"):
                united_keys.append(key)
            for key in profile_to_compare.get("freq"):
                if key not in united_keys:
                    united_keys.append(key)
            for key in united_keys:
                if key in unknown_profile["freq"]:
                    unknown_profile_frequency.append(unknown_profile["freq"].get(key))
                else:
                    unknown_profile_frequency.append(0)
                if key in profile_to_compare["freq"]:
                    profile_to_compare_frequency.append(profile_to_compare["freq"].get(key))
                else:
                    profile_to_compare_frequency.append(0)
            return calculate_mse(unknown_profile_frequency, profile_to_compare_frequency)

    def detect_language(
            unknown_profile: dict[str, str | dict[str, float]],
            profile_1: dict[str, str | dict[str, float]],
            profile_2: dict[str, str | dict[str, float]],
    ) -> str | None:
        """
        Detects the language of an unknown profile
        :param unknown_profile: a dictionary of a profile to determine the language of
        :param profile_1: a dictionary of a known profile
        :param profile_2: a dictionary of a known profile
        :return: a language
        """
        if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) or not isinstance(profile_2, dict):
            return None
        else:
            mse_1 = compare_profiles(unknown_profile, profile_1)
            mse_2 = compare_profiles(unknown_profile, profile_2)
            lang_1 = str(profile_1["name"])
            lang_2 = str(profile_2["name"])
            if mse_1 > mse_2:
                return lang_2
            elif mse_1 < mse_2:
                return lang_1
            else:
                list_lang = ("lang_1", "lang_2")
                list_lang.sort()
                return list_lang[0]


if __name__ == "__main__":
    main()
