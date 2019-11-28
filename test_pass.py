import pytest
import numpy as np
from layers import BatchNorm

def test1():
	'''
	Test batch norm forward pass with an example input and beta=0.
	'''
	X = np.array([[1, 2, 3], 
				  [3, 4, 5]])
	bn = BatchNorm(3)
	Y = bn.forward(X)
	assert np.allclose(Y, np.array([[-1, -1, -1],
									[1, 1, 1]]))

def test2():
	'''
	Test batch norm forward pass with nonzero beta.
	'''
	X = np.array([[1, 2, 3], 
				  [3, 4, 5]])
	bn = BatchNorm(3)
	bn.beta = np.array([1, 2, 3])
	Y = bn.forward(X)
	print(Y)
	assert np.allclose(Y, np.array([[0, 1, 2],
									[2, 3, 4]]))

def test3():
	'''
	Test batch norm backward pass with nonzero beta.
	Comparing computing gradients to expected gradients that are manually calculated.
	'''
	X = np.array([[1, 2, 3], 
				  [3, 4, 5]])
	bn = BatchNorm(3)
	bn.beta = np.array([1, 2, 3])
	Y = bn.forward(X)

	dY = np.array([[-1, 0, 1],
				   [1, 2, -1]])
	dX, [(beta, dbeta)] = bn.backward(dY)
	dX_true = np.array([[-0.5, 0, 0.5],
				   		[0.5, 1, -0.5]])
	dbeta_true = np.array([0, 2, 0])

	assert np.allclose(dbeta, dbeta_true)
	assert np.allclose(dX, dX_true)