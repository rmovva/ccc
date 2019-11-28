import pytest
import numpy as np
from layers import BatchNorm

def test1():
	'''
	Compute numerical gradient dLdX using dLdY, and compare to calculated value of dLdX.
	We use epsilon = 0.0001 as our small input perturbation.
	'''
	bn = BatchNorm(3)
	X = np.array([[1, 2, 3],
				  [3, 4, 5]])

	eps = 0.0001*np.random.random(X.shape)
	X_plus = X + eps
	X_minus = X - eps

	Y_plus = bn.forward(X_plus)
	Y_minus = bn.forward(X_minus)

	dYdX = (Y_plus - Y_minus) / (2*eps)

	dY = np.ones_like(X)
	dX_numerical = dY*dYdX

	dX, [(_, _)] = bn.backward(dY)
	assert np.allclose(dX, dX_numerical)
