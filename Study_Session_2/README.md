# Session 2

**Date of session** : 23rd March 2021

## Objectives of the session:

1. Introduction to basic elements and features of python

- Int
- Float
- String
- Mathematical ops
- if-else and elif
- List
- Tuples
- Inporting Modules
- Loop

2. Simulating Law of large numbers using Python

- random module
- unbiased and biased coin toss


## Session Notebook:

[Introduction to python (static preview)](https://nbviewer.jupyter.org/github/S4DS-IEM/Study-Group/blob/main/Study_Session_2/notebooks/Introduction_to_Python.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S4DS-IEM/Study-Group/blob/main/Study_Session_2/notebooks/Introduction_to_Python.ipynb)

## A breif Summary of the session

### Introduction to Python

Python is an interpreted, object-oriented, high-level programming language. It is simple and easy to learn.Python is dynamically typed language. It supports all six basic operations (`+`, `-`, `/`, `//`, `*` ,`**`) on numbers. It has special type of indexing - negative index which is exclusive in Python. Python also offers string and list. Slice is used get a substring with some parameters. Functions are defined with `def` keyword.lists are use to store any type of data. `append()` and `remove()` are used to insert and remove a data from list.

### Probability simluation

Python is backed up a lot of packages. Packages offer a set of predefined classes and functions that make our task at hand easier. To use these  aweseom pre-defined functions and classes we have to first `import` the module where it is defined.

`random` is a standard library in python. It has functions like uniform which samples from numbers unfiormly from a given range.

<p align="center">
<img src="https://user-images.githubusercontent.com/55111154/119544982-e68f2780-bdaf-11eb-8044-f67acf5c3ccb.png" width=500 >
</p>

We write a function to simulate the proability of a unbiased coin

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
Function to simulate the probability of a biased coin where `p` is the probability of heads.

```python
def biased(p):

  x=random.uniform(0,1)

  if x>(1-p):
     return True
  else :
     return False
```

Toss the coin `N` of times and record the probability of the coin.

```python
N = 10
results = []

# Toss the coin 10 times and store that in a list
for i in range(N):
    result = unbiased_coin_toss()
    results.append(result)

n_heads = sum(results)
p_heads = n_heads/N

print("Probability is {:.2f}".format(p_heads))
```
----
The summary was written by [Chaitak Gorai](https://github.com/chaitak-gorai) and reviewed by [Deeptendu Santra](https://github.com/Dsantra92).


