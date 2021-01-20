# import <package name>
import random

# Print without newline
print("Hello World", end ="")

# Print with new line
print("Hello World")

for i in range(0, 10):
    print(i)


def square(a):
    return a*a


# Call the function
print(square(6))

# Vectorization 
import numpy as np # Vectorization is possible in numpy
sq = np.vectorize(square)
vec_sq = sq([1, 2, 3, 4, 5])

# Mapping
mapped_vec = list(map(lambda x : square(x), [1, 2, 3, 4, 5]))