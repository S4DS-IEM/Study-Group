# Session 2

**Date of session** : 23rd March 2021

Objectives of the session:

1. Starting with Python

- Int
- Float
- String
- List
- Tuples
- Random Module
- Loop

2.Introduction to Probability with Python

- random module
- nbiased and biased coin toss
- Graph

Session Notebook:

[Introduction to python (static preview)](https://nbviewer.jupyter.org/github/S4DS-IEM/Study-Group/blob/main/Study_Session_2/notebooks/Introduction_to_Python.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S4DS-IEM/Study-Group/blob/main/Study_Session_2/notebooks/Introduction_to_Python.ipynb)

# Python

Python is an interpreted, object-oriented, high-level programming language.It is simple and easy to learn.Python is dynamically typed language.It has basic six types of operations (+,-,/,//,\* ,\***). It has special type of indexing - negative index which is exclusive only in Python. It has strings and lists. Slice is used get a substring with some parameters. Functions are defined with **def** keyword.lists are use to store any type of data. **append()** and **remove()** are used to insert and remove a data from list.

# Probability

To use some already defined functions we use pakages. For this we have to first import the module. **random** is a type of module.It has functions like uniform(which gives a random no in a range.A unbiase event is a event where all possible outcomes are equally likely and it is called unbiased when the possible outcomes are not equally likely.
Below down the sample code for biased and unbiased coin toss.



```Python
def unbiased_coin_toss():
    # Generate a random number from 0 to 1
    x = random.uniform(0, 1)
    # Probability that the number falls between 0 and 0.5 is 1/2

    if x > 0.5:
        return True
    else:
        return False
```
```python
def biased(h):

  x=random.uniform(0,1)

  if x>(1-h):
     return True
  else :
     return False
```



