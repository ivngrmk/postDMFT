from typing import Any
import numpy as np
from scipy import linalg


class TrygPoly():
    """ Class to represent trigonometric polynomial as a function in R^dim space. """
    def __init__(self, coefficients):
        """
        Args:
            coefficients (ndarray of shape (2*n1+1,2*n2+1,...) or even (2*n1+1,) in 1D case): ni - number of positive frequencies along ith axis. 
        """
        self.coefficients = np.array(coefficients)
        # This array stores ni for i=1, ..., dim .
        self.coefficients_shape = (
            np.array(coefficients.shape, dtype=int) - 1) // 2
        self.dim = len(self.coefficients_shape)

    def __call__(self, x_in):
        # If dim == 1 than x_in is a number converted to a numpy array x = np.array([x]) .
        # Otherwise x_in is expected to be numpy array and x = x_in.
        x = scalar2array(x_in)
        multi_indeces = np.indices(self.coefficients.shape) - np.reshape(self.coefficients_shape,[len(self.coefficients_shape),]+[1,]*len(self.coefficients.shape))
        exponents  = -1j*np.sum(multi_indeces*np.reshape(x,[len(x),]+[1]*(len(multi_indeces.shape)-1)),axis=0)
        value = np.sum(self.coefficients*np.exp(exponents))
        return value

class TrygPoly2DRealPart(TrygPoly):
    def __call__(self, x, y):
        return np.real(super().__call__(np.array([x,y])))
    
class TrygPoly2DImagPart(TrygPoly):
    def __call__(self, x, y):
        return np.imag(super().__call__(np.array([x,y])))


class FourierApproximation():
    """ Class to represent fourier approximation mode.
    """
    def __init__(self, input_mesh, input_values, coefficients_shape):
        """
        Args:
            input_mesh (list of ndarray point)
            input_values (list or ndarray of numbers)
            coefficients_shape (list of ni for i=1,...,dim, see TrygPoly for details. )
        """
        self.mesh = input_mesh
        self.data = input_values
        self.coefficients_shape = scalar2array(coefficients_shape)
        self.total_coefficient_space_size = np.prod(self.coefficients_shape * 2 + 1)
        self.coefficients = np.zeros(
            self.coefficients_shape*2+1, dtype=complex)
        self.dim = len(self.coefficients_shape)
        if self.dim == 1:
            self.mesh = self.mesh[:,None]
        self.compute_approximant()

    def get_F(self):
        total_size = self.total_coefficient_space_size
        F = np.zeros(total_size, dtype=complex)
        k = -1
        coeff_it = np.nditer(self.coefficients, flags=['multi_index'])
        for _ in coeff_it:
            k += 1
            multi_index = coeff_it.multi_index
            frequencies = multi_index - self.coefficients_shape
            F[k] = np.sum(self.data*np.exp(np.matmul(self.mesh,frequencies)*1j))
        return F

    def get_K(self):
        total_size = self.total_coefficient_space_size
        K = np.zeros((total_size, total_size), dtype=complex)
        l = -1
        coeff_it_l = np.nditer(self.coefficients, flags=['multi_index'])
        for _ in coeff_it_l:
            l += 1
            multi_index_l = coeff_it_l.multi_index
            frequencies_l = multi_index_l - self.coefficients_shape
            r = -1
            coeff_it_r = np.nditer(self.coefficients, flags=['multi_index'])
            for _ in coeff_it_r:
                r += 1
                multi_index_r = coeff_it_r.multi_index
                frequencies_r = multi_index_r - self.coefficients_shape
                K[l,r] = np.sum(np.exp(np.matmul(self.mesh,frequencies_l-frequencies_r)*1j))
        return K

    def compute_approximant(self):
        K = self.get_K()
        F = self.get_F()
        self.coefficients = linalg.solve(
            K, F).reshape(self.coefficients_shape*2+1)
        self.approximant = TrygPoly(self.coefficients)

    def __call__(self, *args: Any, **kwds: Any):
        return self.approximant(*args, **kwds)


def scalar2array(x):
    """ Function to convert from a number to 1D ndarray of shape (1,) in the 1D case."""
    dtype = np.array(x).dtype
    if not np.array(x).shape:
        return np.array([x],dtype=dtype)
    else:
        return np.array(x,dtype=dtype)
    
class FourierApproximation2D(FourierApproximation):
    def __init__(self, X, Y, data, nx, ny):
        mesh = np.zeros((len(X)*len(Y),2),dtype=float)
        k = -1
        for x in X:
            for y in Y:
                k += 1
                mesh[k] = np.array([x,y])
        data_converted = data.flatten()
        super().__init__(mesh, data_converted, np.array((nx,ny),dtype=int))

    def __call__(self, *args: Any, **kwds: Any):
        return self.approximant(*args, **kwds)