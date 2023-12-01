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

    json_paths = ['assets/profiles/de.json',
                  'assets/profiles/en.json',
                  'assets/profiles/es.json',
                  'assets/profiles/fr.json',
                  'assets/profiles/it.json',
                  'assets/profiles/ru.json',
                  'assets/profiles/tr.json']


    if __name__ == "__main__":
        main()
import main
f = open('assets/texts/en.txt', 'r')
f_text = f.read()
print(main.tokenize(f_text))

