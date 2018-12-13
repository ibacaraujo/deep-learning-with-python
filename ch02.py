import numpy as np

# Naive ReLU
def naive_relu(x):
  res = x.copy()
  for index, i in enumerate(x):
    res[index] = max(i, 0) 
  return res
	
print(naive_relu([1, 2, 3, 4, -4, -3, -2, -1]))

def naive_relu_2d(x):
  assert len(x.shape) == 2
	
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] = max(x[i, j], 0)
  return x

print(naive_relu_2d(np.array([[1, 2, 3, 4], 
			      [-4, -3, -2, -1]])))

# Addition
def naive_addition(x, y):
  assert len(x) == len(y)
	
  x = x.copy()
  for i in range(len(x)):
    x[i] = x[i] + y[i]
  return x
  
print(naive_addition([1, 2, 3, 4], [1, 2, 3, 4]))

def naive_addition_2d(x, y):
  assert len(x.shape) == 2
  assert x.shape == y.shape
	
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] = x[i, j] + y[i, j]
  return x
  
print(naive_addition_2d(np.array([[1, 2], [3, 4]]), 
						np.array([[1, 2], [3, 4]])))

def naive_add_matrix_and_vector(x, y):
	assert len(x.shape) == 2 # 2D Numpy tensor
	assert len(y.shape) == 1 # 1D Numpy tensor
	assert x.shape[1] == y.shape[0]
	
	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i, j] += y[j]
	return x
	
print(naive_add_matrix_and_vector(np.array([[1, 2], [3, 4]]), np.array([1, 2])))
