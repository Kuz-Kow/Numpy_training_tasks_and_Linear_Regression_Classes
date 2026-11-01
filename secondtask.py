import numpy as np


X = np.array([
    [180, 80, 25],
    [165, 60, 30],
    [175, 75, 22],
    [190, 90, 35],
    [170, 65, 28],
])
mean = np.mean(X, axis = 0)
str_format = "Height: {}, Weight: {}, Age: {}"

#print(str_format.format(*X.mean(axis = 0)))
#print(np.std(X, axis = 0))

#X_standatized = ((X - mean) / np.std(X, axis = 0))
#print(X_standatized)

index = np.argmax(X[:, 1])
print(X[index])
older_people = X[X[:, 2] > 25]
print(older_people)