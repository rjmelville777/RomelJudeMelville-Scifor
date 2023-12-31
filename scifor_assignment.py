# -*- coding: utf-8 -*-
"""Scifor Assignment

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IG7hEB7dH-WZO3_H4dcfggiWKPg2-lsj
"""

Q1. Create a python program to sort the given list of tuples based on integer value using a lambda function.
[('Sachin Tendulkar', 34357), ('Ricky Ponting', 27483), ('Jack Kallis', 25534), ('Virat Kohli', 24936)]

cricket = [('Sachin Tendulkar', 34357), ('Ricky Ponting', 27483), ('Jack Kallis', 25534), ('Virat Kohli', 24936)]

sorted_data = sorted(cricket, key=lambda x: x[1])

for item in sorted_data:
    print(item)

Q2. Write a Python Program to find the squares of all the numbers in the given list of integers using lambda and map functions.
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = list(map(lambda x: x**2, numbers))
print(squares)

Q3. Write a python program to convert the given list of integers into a tuple of strings. Use map and lambda functions
Given String: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Expected output: ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tuple_of_strings = tuple(map(lambda x: str(x), numbers))
print(tuple_of_strings)

Q4. Write a python program using reduce function to compute the product of a list containing numbers from 1 to 25.

from functools import reduce
numbers = list(range(1, 26))

def multiply(x, y):
    return x * y

product = reduce(multiply, numbers)
print("The product of the numbers from 1 to 25 is:", product)

Q5. Write a python program to filter the numbers in a given list that are divisible by 2 and 3
 [2, 3, 6, 9, 27, 60, 90, 120, 55, 46]

numbers = [2, 3, 6, 9, 27, 60, 90, 120, 55, 46]

filtered_numbers = [num for num in numbers if num % 2 == 0 and num % 3 == 0]

print("Numbers divisible by 2 and 3:", filtered_numbers)

Q6. Write a python program to find palindromes in the given list of strings using lambda and filter function.
['python', 'php', 'aba', 'radar', 'level']

strings = ['python', 'php', 'aba', 'radar', 'level']
palindromes = list(filter(lambda x: x == x[::-1], strings))
print(palindromes)



