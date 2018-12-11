import numpy as np

# Naive ReLU
def naive_relu(x):
	res = x.copy()
	for index, i in enumerate(x):
		res[index] = max(i, 0) 
	return res
	
print(naive_relu([1, 2, 3, 4, -4, -3, -2, -1]))
