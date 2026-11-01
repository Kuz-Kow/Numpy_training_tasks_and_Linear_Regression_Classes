import numpy as np


data = np.random.rand(5, 5)

X = data[:, :4]
y = data[:, 4]



X_standartized = (X - X.mean(axis=0)) / X.std(axis= 0)


y_standatized = (y - y.mean(axis = 0))/ y.std(axis = 0)

weights = np.random.rand(4)

weights = (weights - weights.mean(axis = 0)) / weights.std(axis = 0)


prediction = X_standartized @ weights


MSE = np.mean((y_standatized - prediction)**2)
print(MSE)