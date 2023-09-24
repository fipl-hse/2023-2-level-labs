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
    return input_string * how_many


# Function calls with expected result:
# multiply_string('Hi', 2)
# multiply_string('Hi', 3)
# multiply_string('Hi', 1)
# multiply_string('Hi', 0)


# Task 2:
def front_times(input_string: str, how_many: int) -> str:
    """
    Given the string, take its three leading characters
    and display them that many times as in `how_many`.
    """
    # student realisation goes here
    return input_string[:3] * how_many


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
    return input_string[-2:] * 3


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
    return first_string + second_string * 2 + first_string


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
    cypher = []
    words = sentence.split()
    for each in words:
        if len(each) >= 5:
            cypher.append(each[::-1])
        else:
            cypher.append(each)
    return ' '.join(cypher)


print(reverse_word("Hey fellow warriors"))
# "Hey wollef sroirraw"
print(reverse_word("This is a test"))
# "This is a test"
print(reverse_word("This is another test"))
# "This is rehtona test")


# Task 6
def generate_hashtag(input_string: str) -> str | False:
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
    if not input_string:
        return False
    hashtag = '#'
    for word in input_string.split():
        hashtag += word.capitalize()
    if not hashtag or len(hashtag) > 140:
        return False
    return hashtag


print(generate_hashtag(" Hello there thanks for trying my quiz"))
print(generate_hashtag("    Hello     World   "))
print(generate_hashtag(""))


# Task 7:
def combo_string(first_string: str, second_string: str) -> str:
    """
    Given two strings, concatenate like the following: shorter+longer+shorter
    """
    # student realisation goes here
    if len(first_string) > len(second_string):
        return second_string + first_string + second_string
    return first_string + second_string + first_string


print(combo_string('Hello', 'hi'))
print(combo_string('hi', 'Hello'))
print(combo_string('aaa', 'b'))
print(combo_string('', 'bb'))
print(combo_string('aaa', '1234'))
print(combo_string('bb', 'a'))
# → 'hiHellohi'
# → 'hiHellohi'
# → 'baaab'
# → 'bb'
# → 'aaa1234aaa'
# → 'abba'


# Task 1: advanced
def string_splosion(input_string: str) -> str:
    """
    Given the string, format it like in the example.
    """
    # student realisation goes here
    plosion = ''
    for i in range(len(input_string)):
        plosion += input_string[:i+1]
    return plosion


print(string_splosion('Code'))
print(string_splosion('abc'))
print(string_splosion('ab'))
print(string_splosion('Kitten'))
print(string_splosion('x'))
# Function calls with expected result:
#  → 'CCoCodCode'
#  → 'aababc'
#  → 'aab'
#  → 'KKiKitKittKitteKitten'
#  → 'x'


# Task 2: advanced
def string_match(first_string: str, second_string: str) -> int:
    """
    Given two strings, find the number of times an arbitrary substring (with length of 2)
    is found at the same position in both strings.
    """
    # student realisation goes here
    counter = 0
    for i in range(min(len(first_string), len(second_string)) - 1):
        if first_string[i:i+2] == second_string[i:i+2]:
            counter += 1
    return counter


print(string_match('xxcaazz', 'xxbaaz'))
print(string_match('abc', 'abc'))
print(string_match('abc', 'axc'))
print(string_match('he', 'hello'))
print(string_match('h', 'hello'))
print(string_match('aabbccdd', 'abbbxxd'))
print(string_match('aaxxaaxx', 'iaxxai'))
print(string_match('iaxxai', 'aaxxaaxx'))
# Function calls with expected result:
#  → 3
# NOTE: «xx», «aa» and «az» are found at the same position in both strings
#  → 2
#  → 0
#  → 1
#  → 0
#  → 1
#  → 3
#  → 3
