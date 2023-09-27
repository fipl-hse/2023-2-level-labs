import string
def tokenize(text):
    if text == str:
        text = text.lower()
        text1 = text.translate(str.maketrans('', '', string.punctuation))
        text2 = text1.replace(" ", "")
        a = list(text2)
        return a
    else:
        return None
tokenize("Hey! How are you?")
def calculate_frequencies(a):
   a = freq = dict((i, a.count(i)) for i in a)
    print(a)
calculate_frequencies()



