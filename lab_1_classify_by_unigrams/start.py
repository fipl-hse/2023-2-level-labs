"""
Language detection starter
"""
# from main import load_profile
'''
I couldn't figure out how to import functions as it was implied in Step 0. 
It seems strange because both the function below and the main file share the same name
 
'''

def main() -> None:
    """
    Launches an implementation
    """
    profiles_paths = ['assets/profiles/de.json',
                      'assets/profiles/en.json',
                      'assets/profiles/es.json',
                      'assets/profiles/fr.json',
                      'assets/profiles/it.json',
                      'assets/profiles/ru.json',
                      'assets/profiles/tr.json',
                      ]
    # language_profiles = []
    # for path in profiles_paths:
    #     language_profiles.append(load_profile(path))

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
