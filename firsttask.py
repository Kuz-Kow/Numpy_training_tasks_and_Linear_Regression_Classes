import numpy as np


rng = np.random.default_rng()

array = rng.random(20)

array = array.reshape(4,5)
#print(array)
#print(array[:, 2])
#for row in range(array.shape[1]):
#    print(f"Row {row} : {array[:, row].mean()}")

#array = array * 100
#print(array[array > 10])

print(array.mean(axis=0))