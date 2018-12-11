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
