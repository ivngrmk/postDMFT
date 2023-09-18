import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import solve


def ancont(X, Y, Na, Nb, X_out, N_it=1, ores=False):
    """ Аналитическое продолжение функцией вида: \frac{sum^{N_a-1}_{n=0} z^n a_n}{sum^{N_b-1}_{n=0} z^n b_n + z^{N_b}}.
        Степень числителя: N_a - 1
        Степень знаменателя: N_b
        Коэффициент при z^{N_b} в знаменателе фиксирован и положен b_{N_b} = 1.
        Затухание на бесконечности обеспечивается при N_b >= N_a .
        Число степеней свободы: N_a (числитель) + N_B (знаменатель).

    Args:
        X (_type_): _description_
        Y (_type_): _description_
        Na (_type_): _description_
        Nb (_type_): _description_
        X_out (_type_): _description_
        N_it (int, optional): Число итераций, используемых в получении МНК решения. Defaults to 1.
        ores (bool, optional): Определяет, будут ли выведены коэффициенты, по которым восстанавливается апроксимант. Defaults to False.
    """
    def pade(a, b, x):
        num = 0.0
        for i in range(len(a)):
            num += a[i]*x**i
        den = 0.0
        for i in range(len(b)):
            den += b[i]*x**i
        return num/den
    Np = len(X)
    N = Na + Nb
    # Стартовое приближение - интерполянт по первым точкам.
    A_lhs = np.zeros((N, N), dtype=complex)
    b_rhs = np.zeros(N, dtype=complex)
    for l in range(N):
        line = l
        for n in range(Na):
            col = n
            A_lhs[line, col] = X[l]**n
        for n in range(Nb):
            col = Na + n
            A_lhs[line, col] = -X[l]**n*Y[l]
    for l in range(N):
        line = l
        b_rhs[line] = X[l]**Nb*Y[l]
    v = solve(A_lhs, b_rhs)
    a = v[:Na]
    b = np.array(list(v[Na:]) + [1.0], dtype=complex)
    wght = np.zeros(Np, dtype=complex)
    for m in range(Np):
        denom = 0.0
        for n in range(Nb+1):
            denom += b[n]*X[m]**n
        wght[m] = 1/abs(denom)**2
    # Итерации для МНК.
    for iter in range(N_it):
        A_lhs = np.zeros((N, N), dtype=complex)
        for l in range(Na):
            lin = l
            for n in range(Na):
                col = n
                summ = 0.0
                for m in range(Np):
                    summ += X[m]**n*np.conj(X[m])**l*wght[m]
                A_lhs[lin, col] = summ
            for n in range(Nb):
                col = Na + n
                summ = 0.0
                for m in range(Np):
                    summ += -X[m]**n*np.conj(X[m])**l*wght[m]*Y[m]
                A_lhs[lin, col] = summ
        for l in range(Nb):
            lin = Na + l
            for n in range(Na):
                col = n
                summ = 0.0
                for m in range(Np):
                    summ += X[m]**n*np.conj(X[m])**l*wght[m]*np.conj(Y[m])
                A_lhs[lin, col] = summ
            for n in range(Nb):
                col = Na + n
                summ = 0.0
                for m in range(Np):
                    summ += -X[m]**n*np.conj(X[m])**l * \
                        wght[m]*np.conj(Y[m])*Y[m]
                A_lhs[lin, col] = summ
        b_rhs = np.zeros(N, dtype=complex)
        for l in range(Na):
            lin = l
            summ = 0.0
            for m in range(Np):
                summ += X[m]**Nb*np.conj(X[m])**l*wght[m]*Y[m]
            b_rhs[lin] = summ
        for l in range(Nb):
            lin = Na + l
            summ = 0.0
            for m in range(Np):
                summ += X[m]**Nb*np.conj(X[m])**l*wght[m]*np.conj(Y[m])*Y[m]
            b_rhs[lin] = summ
        v = lstsq(A_lhs, b_rhs, rcond=None)[0]
        a = v[:Na]
        b = np.array(list(v[Na:]) + [1.0], dtype=complex)
        for m in range(Np):
            denom = 0.0
            for n in range(Nb+1):
                denom += b[n]*X[m]**n
            wght[m] = 1/abs(denom)**2
    Y_out = np.zeros(len(X_out), dtype=complex)
    for i, x in enumerate(X_out):
        Y_out[i] = pade(a, b, x)
    if ores:
        res = 0.0
        for m in range(Np):
            res += abs(pade(a, b, X[m])-Y[m])
        res /= Np
        return Y_out, res
    else:
        return Y_out


def ancont_asmp(X_in, Y_in, s, N_in, X_out, N_it=0):
    """ Аналитическое продолжение функцией вида: f(z) = s[0] + s[1]*\frac{z^N_in + sum^{N_in-1}_{n=0} z^n a_n}{sum^{N_in}_{n=0} z^n b_n + z^(N+1)},
        что обеспечивает поведение на бесконечности вида: f(z) ~ s[0] + s[1] / z .
        Степень числителя: N_in
        Степень знаменателя: N_in+1
        Число степеней свободы: N_in (числитель) + (N_in + 1) (знаменатель).

    Args:
        X_in (_type_): _description_
        Y_in (_type_): _description_
        s (_type_): tuple of a form [s[0],s[1]], where s[0] = \lim_{z -> +\infty} f(z), s[1] = \lim_{z -> \infty} (z*f(z) - s[0]*z)
        N_in (_type_): _description_
        X_out (_type_): _description_
        N_it (int, optional): Число итераций, используемых в получении МНК решения. Defaults to 0.
    """
    if N_it != 0:
        print("WARNING!!! Not checked.")

    def pade(a, b, x):
        num = 0.0
        for i in range(len(a)):
            num += a[i]*x**i
        num += x**len(a)
        den = 0.0
        for i in range(len(b)):
            den += b[i]*x**i
        return num/den*s[1]+s[0]
    X = X_in.copy()
    Y = np.zeros(len(Y_in), dtype=complex)
    Np = len(X)
    Na = N_in
    Nb = N_in+1
    N = Na + Nb
    for i, y in enumerate(Y_in):
        Y[i] = (Y_in[i]-s[0])/s[1]
    # Стартовое приближение - интерполянт по первым точкам.
    A_lhs = np.zeros((N, N), dtype=complex)
    b_rhs = np.zeros(N, dtype=complex)
    for l in range(N):
        line = l
        for n in range(Na):
            col = n
            A_lhs[line, col] = X[l]**n
        for n in range(Nb):
            col = Na + n
            A_lhs[line, col] = -X[l]**n*Y[l]
    for l in range(N):
        line = l
        b_rhs[line] = X[l]**Nb*Y[l]-X[l]**(Nb-1)
    v = solve(A_lhs, b_rhs)
    a = v[:Na]
    b = np.array(list(v[Na:]) + [1.0], dtype=complex)
    wght = np.zeros(Np, dtype=complex)
    for m in range(Np):
        denom = 0.0
        for n in range(Nb+1):
            denom += b[n]*X[m]**n
        wght[m] = 1/abs(denom)**2
    # Итерации для МНК.
    for iter in range(N_it):
        A_lhs = np.zeros((N, N), dtype=complex)
        for l in range(Na):
            lin = l
            for n in range(Na):
                col = n
                summ = 0.0
                for m in range(Np):
                    summ += X[m]**n*np.conj(X[m])**l*wght[m]
                A_lhs[lin, col] = summ
            for n in range(Nb):
                col = Na + n
                summ = 0.0
                for m in range(Np):
                    summ += -X[m]**n*np.conj(X[m])**l*wght[m]*Y[m]
                A_lhs[lin, col] = summ
        for l in range(Nb):
            lin = Na + l
            for n in range(Na):
                col = n
                summ = 0.0
                for m in range(Np):
                    summ += X[m]**n*np.conj(X[m])**l*wght[m]*np.conj(Y[m])
                A_lhs[lin, col] = summ
            for n in range(Nb):
                col = Na + n
                summ = 0.0
                for m in range(Np):
                    summ += -X[m]**n*np.conj(X[m])**l * \
                        wght[m]*np.conj(Y[m])*Y[m]
                A_lhs[lin, col] = summ
        b_rhs = np.zeros(N, dtype=complex)
        for l in range(Na):
            lin = l
            summ = 0.0
            for m in range(Np):
                summ += (X[m]**Nb-X[m]**(Nb-1)/Y[m]) * \
                    np.conj(X[m])**l*wght[m]*Y[m]
            b_rhs[lin] = summ
        for l in range(Nb):
            lin = Na + l
            summ = 0.0
            for m in range(Np):
                summ += (X[m]**Nb-X[m]**(Nb-1)/Y[m])**Nb * \
                    np.conj(X[m])**l*wght[m]*np.conj(Y[m])*Y[m]
            b_rhs[lin] = summ
        v = lstsq(A_lhs, b_rhs, rcond=None)[0]
        a = v[:Na]
        b = np.array(list(v[Na:]) + [1.0], dtype=complex)
        for m in range(Np):
            denom = 0.0
            for n in range(Nb+1):
                denom += b[n]*X[m]**n
            wght[m] = 1/abs(denom)**2
    Y_out = np.zeros(len(X_out), dtype=complex)
    for i, x in enumerate(X_out):
        Y_out[i] = pade(a, b, x)
    return Y_out
