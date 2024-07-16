import numpy as np
from numpy.polynomial.polynomial import Polynomial, polydiv
from scipy.stats import norm as normal_distribution
from scipy.optimize import minimize

def spin2index(symbol):
    if symbol == 'x':
        return 0
    if symbol == 'y':
        return 1
    if symbol == 'z':
        return 2
    raise TypeError("Unrecognized symbol: " + str(symbol))

def index2spin(m):
    if m == 0:
        return 'x'
    if m == 1:
        return 'y'
    if m == 2:
        return 'z'
    raise TypeError("Unrecognized number: " + str(m))

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