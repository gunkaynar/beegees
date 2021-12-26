import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
	return max(0.0, x)

def sigmoid_backward(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def relu_backward(x):
    return (np.greater(x, 0).astype(int))