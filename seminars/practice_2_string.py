"""
Programming 2022
Seminar 2


Data Type: String
"""

# pylint: disable=invalid-name

# Common information about strings
#
# strings are immutable
# strings are iterable
# strings are case-sensitive

# Create a string
example = 'Hello'  # or "Hello"
print(example)

# String concatenation
greeting = example + ' there!'
print(greeting)

# String multiplication
several_hellos = example * 5
print(several_hellos)

# String formatting
# .format() method
example = '{} there!'.format(example)  # pylint: disable=consider-using-f-string
print(example)
# f-strings
example = f'{greeting} - this is the "greeting" variable.'
print(example)


# String methods (some of them)
# .split() -> split by the delimiter
# .join() - join by the delimiter
# .upper() - uppercase copy of the string
# .lower() - lowercase copy of the string
# .isalpha() - if all the characters in the text are letters
# .strip() - remove the given element (space by default) from the both ends of the string
# .find() - search the string for the specified value (return the index of the first occurrence)

# TASKS

# Task 1:
def multiply_string(input_string: str, how_many: int) -> str:
    """
    Given a string and a non-negative number,
    display the given string the number of times given in the `how_many`.
    """
    # student realisation goes here
    format_string = input_string * how_many
    print(format_string)
    return format_string


# Function calls with expected result:
# multiply_string('Hi', 2) → 'HiHi'
# multiply_string('Hi', 3) → 'HiHiHi'
# multiply_string('Hi', 1) → 'Hi'
# multiply_string('Hi', 0) → ''


# Task 2:
def front_times(input_string: str, how_many: int) -> str:
    """
    Given the string, take its three leading characters
    and display them that many times as in `how_many`.
    """
    # student realisation goes here
    if len(input_string) >= 3:
        first_characters = input_string[:4]
    else:
        first_characters = input_string[:]
    print(first_characters * how_many)
    return first_characters * how_many


front_times('Chocolate', 3)
# Function calls with expected result:
# front_times('Chocolate', 2) → 'ChoCho'
# front_times('Chocolate', 3) → 'ChoChoCho'
# front_times('Abc', 3) → 'AbcAbcAbc'
# front_times('A', 4) → 'AAAA'
# front_times('', 4) → ''
# front_times('Abc', 0) → ''


# Task 3:
def extra_end(input_string: str) -> str:
    """
    Given the string, take its two last characters and display them three times.
    """
    # student realisation goes here
    print(input_string[-2:] * 3)
    return input_string[-2] * 3


extra_end('Hello')
# Function calls with expected result:
# extra_end('Hello') → 'lololo'
# extra_end('ab') → 'ababab'
# extra_end('Hi') → 'HiHiHi'
# extra_end('Code') → 'dedede'


# Task 4:
def make_abba(first_string: str, second_string: str) -> str:
    """
    Given two strings, concatenate them as a reflection.
    """
    # student realisation goes here
    string_common = first_string + second_string
    common_string = second_string + first_string
    print(string_common + common_string)
    return string_common + common_string


make_abba('Hi', 'Bye')
# make_abba('Hi', 'Bye') → 'HiByeByeHi'
# make_abba('Yo', 'Alice') → 'YoAliceAliceYo'
# make_abba('What', 'Up') → 'WhatUpUpWhat'
# make_abba('', 'y') → 'yy'


# Task 5
def reverse_word(sentence: str) -> str:
    """
    Write a function that takes in a string of one or more words,
    and returns the same string, but with all five or more letter words reversed.

    Strings passed in will consist of only letters and spaces.
    Spaces will be included only when more than one word is present.
    """
    # student realisation goes here
    sentence = sentence.split()
    new_sentence = []
    for word in sentence:
        if len(word) < 5:
            new_sentence.append(word)
        else:
            new_sentence.append(word[::-1])
    new_sentence = ' '.join(new_sentence)
    print(new_sentence)
    return new_sentence


reverse_word('Hey fellow warriors')
# reverse_word("Hey fellow warriors") == "Hey wollef sroirraw"
#
# reverse_word("This is a test") == "This is a test"
#
# reverse_word("This is another test") == "This is rehtona test"


# Task 6
def generate_hashtag(input_string: str) -> str:
    """
    The marketing team is spending way too much time typing in hashtags.
    Let's help them with our own Hashtag Generator!

    Here's the deal:

    It must start with a hashtag (#).
    All words must have their first letter capitalized.
    If the final result is longer than 140 chars it must return false.
    If the input or the result is an empty string it must return false.
    Examples
    " Hello there thanks for trying my quiz"  =>  "#HelloThereThanksForTryingMyQuiz"
    "    Hello     World   "                  =>  "#HelloWorld"
    ""                                        =>  false
    """
    # student realisation goes here
    if len(input_string) == 0:
        return 'false'
    hashtag_list = input_string.split()
    new_hashtag = ['#']
    for word in hashtag_list:
        new_hashtag.append(word.title())
    new_hashtag = ''.join(new_hashtag)
    if len(new_hashtag) > 140:
        return 'false'
    print(new_hashtag)
    return new_hashtag


generate_hashtag('   Hello there thanks    for trying my quiz   ')


# Task 7:
def combo_string(first_string: str, second_string: str) -> str:
    """
    Given two strings, concatenate like the following: shorter+longer+shorter
    """
    # student realisation goes here
    string_list = [first_string, second_string]
    string_list = sorted(string_list, key=len)
    print(string_list[0] + string_list[1] + string_list[0])
    return string_list[0] + string_list[1] + string_list[0]


combo_string('Hello', 'Hi')
# combo_string('Hello', 'hi') → 'hiHellohi'
# combo_string('hi', 'Hello') → 'hiHellohi'
# combo_string('aaa', 'b') → 'baaab'
# combo_string('', 'bb') → 'bb'
# combo_string('aaa', '1234') → 'aaa1234aaa'
# combo_string('bb', 'a') → 'abba'


# Task 1: advanced
def string_splosion(input_string: str) -> str:
    """
    Given the string, format it like in the example.
    """
    # student realisation goes here
    new_string = []
    n = 0
    while n < len(input_string):
        new_string.append(input_string[:n + 1])
        n += 1
    print(''.join(new_string))
    return ''.join(new_string)


string_splosion('Code')
# Function calls with expected result:
# string_splosion('Code') → 'CCoCodCode'
# string_splosion('abc') → 'aababc'
# string_splosion('ab') → 'aab'
# string_splosion('Kitten') → 'KKiKitKittKitteKitten'
# string_splosion('x') → 'x'


# Task 2: advanced
def string_match(first_string: str, second_string: str) -> int:
    """
    Given two strings, find the number of times an arbitrary substring (with length of 2)
    is found at the same position in both strings.
    """
    # student realisation goes here
    substring_list1 = []
    n = 0
    while n < len(first_string) - 1:
        substring_list1.append(first_string[n:n + 2])
        n += 1
    substring_list2 = []
    n = 0
    while n < len(first_string) - 1:
        substring_list2.append(second_string[n:n + 2])
        n += 1
    substring_common = set(substring_list2).intersection(set(substring_list1))
    print(len(substring_common))
    return len(substring_common)


string_match('xxcaazz', 'xxbaaz')
# Function calls with expected result:
# string_match('xxcaazz', 'xxbaaz') → 3
# NOTE: «xx», «aa» and «az» are found at the same position in both strings
# string_match('abc', 'abc') → 2
# string_match('abc', 'axc') → 0
# string_match('he', 'hello') → 1
# string_match('h', 'hello') → 0
# string_match('aabbccdd', 'abbbxxd') → 1
# string_match('aaxxaaxx', 'iaxxai') → 3
# string_match('iaxxai', 'aaxxaaxx') → 3
