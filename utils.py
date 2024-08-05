import numpy as np
from numpy.polynomial.polynomial import Polynomial, polydiv
from scipy.stats import norm as normal_distribution
from scipy.optimize import minimize
from postDMFT.common import FLOATZERO

def list1d_to_dict(list1d: list,strings: list) -> dict:
    """Convertes 1D list to a dictionary.

    Args:
        list1d (list): 1D list.
        strings (list): List of symbols.

    Raises:
        RuntimeError: Is the size of list1d does not match the size of strings.

    Returns:
        dict: Dictionary with keys 'A', where A is a symbol from strings list.
    """
    if len(list1d) != len(strings):
        raise RuntimeError(f"{len(list1d)},{len(strings)}")
    else:
        dictarray = {}
        for array_idx, array in enumerate(list1d):
            dictarray[strings[array_idx]] = array
        return dictarray
    
def list2d_to_dict(list2d: list,strings: list) -> dict:
    """Convertes nested square 2D list to a dictionary.

    Args:
        list2d (list): Nested square 2D list.
        strings (list): List of symbols.

    Raises:
        RuntimeError: If the list2d list is not NxN or len(strings) != N.

    Returns:
        dict: Dictionary with keys 'AB', where A and B are symbols from strings list.
    """
    is_ok = True
    is_ok = is_ok and (len(list2d) == len(strings))
    for idx in range(len(list2d)):
        is_ok = is_ok and (len(list2d[idx]) == len(strings))
    if not is_ok:
        raise RuntimeError
    dictarray = {}
    for i in range(len(strings)):
        for j in range(len(strings)):
            key = f"{strings[i]}{strings[j]}"
            dictarray[key] = list2d[i][j]
    return dictarray

def remove_small_numbers(array,SMALL_NUMBER=FLOATZERO):
    with np.nditer(array,op_flags=['readwrite']) as it:
        for x in it:
            xre = np.real(x)
            xim = np.imag(x)
            if abs(xre) < SMALL_NUMBER:
                xre = 0.0
            if abs(xim) < SMALL_NUMBER:
                xim = 0.0
            x[...] = complex(xre,xim)

def get_diag(array_2d: np.ndarray):
    if (len(array_2d.shape) != 2 or array_2d.shape[0] != array_2d.shape[1]): raise RuntimeError
    npoints = array_2d.shape[0]
    diag = np.array([array_2d[i,i] for i in range(npoints)])
    return diag

def get_contrdiag(array_2d: np.ndarray):
    if (len(array_2d.shape) != 2 or array_2d.shape[0] != array_2d.shape[1]): raise RuntimeError
    npoints = array_2d.shape[0]
    contrdiag = np.array([array_2d[i,-i-1] for i in range(npoints)])
    return contrdiag

def spin2index(symbol):
    if symbol == 'x':
        return 0
    if symbol == 'y':
        return 1
    if symbol == 'z':
        return 2
    raise TypeError("Unrecognized symbol: " + str(symbol))

# def index2spin(m):
    # if m == 0:
        # return 'x'
    # if m == 1:
        # return 'y'
    # if m == 2:
        # return 'z'
    # raise TypeError("Unrecognized number: " + str(m))

def index2rot(m):
    if m == 0:
        return '+'
    if m == 1:
        return 'y'
    if m == 2:
        return '-'
    if m == 3 or m == -1:
        return '0'
    raise TypeError("Unrecognized number: " + str(m))

def m2str(m):
    if m == 0:
        return "\\uparrow \\uparrow"
    if m == 1:
        return "\\uparrow \\downarrow"
    if m == 2:
        return "\\downarrow \\uparrow"
    if m == 3:
        return "\\downarrow \\downarrow"


def m2spinstr(m):
    if m == 0:
        return "x"
    if m == 1:
        return "y"
    if m == 2:
        return "z"
    if m == 3 or m == -1:
        return "0"
    raise TypeError("Unrecognized number: " + str(m))

index2spin = m2spinstr


def sigma2str(sigma):
    if sigma == 0:
        return "\\uparrow"
    if sigma == 1:
        return "\\downarrow"

class PadeApproximant():
    """ Class for a real-valued Pade Approximant.
    """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.num = Polynomial([0.0*1j,] * (n + 1))
        self.den = Polynomial([0.0*1j,] * m)
        self.den.coef[-1] = 1.0

    def fit_data(self,X,Y,x0=None):
        def fit_f(x,parameters):
            num = Polynomial(parameters[:self.n+1])
            if len(parameters) == self.n + self.m + 1:
                den = Polynomial(list(parameters[self.n+1:]) + [1.0 + 0.0*1j])
            else:
                raise RuntimeError
            return num(x) / den(x)
        def loss_function(parameters):
            loss = 0.0
            for i,x in enumerate(X):
                loss += abs(Y[i] - fit_f(x,parameters))**2
            return loss
        if not x0:
            random_x0 = []
            for _ in range(self.n + self.m + 1):
                random_distr = normal_distribution(loc=0.0,scale=1.0)
                random_x0.append(random_distr.rvs())
            parameters_opt = minimize(loss_function, x0=random_x0)
        else:
            parameters_opt = minimize(loss_function, x0=x0)
        self.num = Polynomial(parameters_opt.x[:self.n+1])
        self.den = Polynomial(list(parameters_opt.x[self.n+1:]) + [1.0 + 0.0*1j])

    def __call__(self,x):
        return self.num(x) / self.den(x)

    def polish(self,threshold=10**(-5)):
        num_roots = self.num.roots()
        den_roots = self.den.roots()
        common_roots = []
        for num_root in num_roots:
            for den_root in den_roots:
                if abs(num_root - den_root) < threshold:
                    common_roots.append(den_root)
        for common_root in common_roots:
            monom = Polynomial([-common_root,1.0])
            self.num = Polynomial(polydiv(self.num.coef,monom.coef)[0])
            self.den = Polynomial(polydiv(self.den.coef,monom.coef)[0])

if __name__ == "__main__":
    print("Imports fine.")