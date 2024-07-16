import numpy as np

FLOATZERO = 10**(-8)

def Fermi(e, T):
    return 1.0/(np.exp(e/T)+1.0)