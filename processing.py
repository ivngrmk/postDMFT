import numpy as np
import ana_cont.continuation as cont
from IPython.utils import io
from scipy import interpolate
from scipy import linalg as lg
from scipy import optimize as opt
import h5py
import scipy

FLOATZERO = 10**(-8)

# Details of ana_cont API and usage can be find: https://josefkaufmann.github.io/ana_cont/api_doc.html and https://arxiv.org/abs/2105.11211 .


def Fermi(e, T):
    return 1/(np.exp(e/T)+1)


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


def v2d(vx, vy):
    """ Function to easily create 2d wave vectors."""
    v = np.zeros(2)
    v[0] = vx
    v[1] = vy
    return v


def vBZ(name):
    """ Function to easily create 2d wave vectors of high-symmetry points in the first Brillouin zone."""
    if name == 'G':
        return v2d(0.0, 0.0)
    elif name == 'M':
        return v2d(np.pi, np.pi)
    elif name == 'X':
        return v2d(np.pi, 0.0)
    elif name == 'S':
        return v2d(np.pi/2, np.pi/2)
    else:
        raise RuntimeError


def periodic(qx_i):
    """ Function to apply periodic conditions on the wave vector component."""
    qx = qx_i
    while qx < -np.pi or np.pi < qx:
        if qx < -np.pi:
            qx += 2*np.pi
        if qx > np.pi:
            qx -= 2*np.pi
    return qx


def map2array1d(func, array: np.ndarray, dtype=complex, answer_shape=None):
    """ Function similar to the map function.
    func: function of 2 arguments (1 2d wave vector);
        this function can be vector-valued or matrix-valued, returning result of a shape: answer_shape;
    dtype: type of the func values;
    array: ndarray of a shape (array.shape[0],2), which represents 1d manifold in the 2d Brillouin zone.
    return: ndarray of a shape (array.shape[0],) + answer_shape."""
    if answer_shape is None:
        answer = np.zeros(array.shape[0], dtype=dtype)
    else:
        answer = np.zeros((array.shape[0],)+answer_shape, dtype=dtype)
    for i, x in enumerate(array):
        answer[i] = func(x[0], x[1])
    return answer


def map2array2d(func, array, dtype=complex, answer_shape=None):
    """ Function similar to the map function.
    func: function of 2 arguments (1 2d wave vector);
        this function can be vector-valued or matrix-valued, returning result of a shape: answer_shape;
    dtype: type of the func values;
    array: ndarray of a shape (array.shape[0],array.shape[1],2), which represents 2d manifold in the 2d Brillouin zone.
    return: ndarray of a shape (array.shape[0],array.shape[1]) + answer_shape."""
    if answer_shape is None:
        answer = np.zeros((array.shape[0], array.shape[1]), dtype=dtype)
    else:
        answer = np.zeros(
            (array.shape[0], array.shape[1])+answer_shape, dtype=dtype)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            x = array[i, j, :]
            answer[i, j] = func(x[0], x[1])
    return answer


def rectangle_mesh(q1: np.ndarray, q2: np.ndarray, nkp_x: int, nkp_y: int):
    """ Function to create 2d grid-mesh on a reсtangle, based on wave vectors q1 and q2.
    nkp_x and nkp_y: number of points along x-axis and along y-axis. 
    returns: ndarray of a shape (nkp_x,nkp_y,2) ."""
    mesh = np.zeros((nkp_x, nkp_y, 2))
    Tx = np.linspace(0.0, 1.0, nkp_x, endpoint=True)
    Ty = np.linspace(0.0, 1.0, nkp_y, endpoint=True)
    for i, tx in enumerate(Tx):
        for j, ty in enumerate(Ty):
            # dim \in {x,y}
            for dim in range(2):
                if dim == 0:
                    t = tx
                else:
                    t = ty
                mesh[i, j, dim] = (q2 - q1)[dim]*t + q1[dim]
    return mesh


def line_mesh(q1: np.ndarray, q2: np.ndarray, nkp: int):
    """ Function to create 1d grid-mesh on an interval, based on wave vectors q1 and q2.
    nkp: number of points along the interval.
    returns: ndarray of a shape (nkp,2) ."""
    mesh = np.zeros((nkp, 2))
    T = np.linspace(0.0, 1.0, nkp, endpoint=True)
    for i, t in enumerate(T):
        # dim \in {x,y}
        for dim in range(2):
            mesh[i, dim] = (q2-q1)[dim]*t + q1[dim]
    return mesh


class BZPath():
    """ Class to manage zig-zag path in the Brillouin zone."""

    def __init__(self, points_list, nkp):
        """ points: list of the points making up the zig-zag path;
                len(points) = self.nint + 1;
                self.nint - number of intervals;
            nkp: number of points on each separate interval. """
        # Number of wave vector points on each interval.
        self.nkp = nkp
        # list of points making up the zig-zag path.
        self.points = points_list
        # number of intervals
        self.nint = len(self.points)-1
        for i in range(self.nint):
            if i == self.nint - 1:
                interval_mesh = line_mesh(
                    self.points[i], self.points[i+1], nkp)
            else:
                # Number of points trick (...,nkp+1)[:-1,:] serves to make shapes of the self.mesh and self.T equal.
                interval_mesh = line_mesh(
                    self.points[i], self.points[i+1], nkp+1)[:-1, :]
            if i == 0:
                self.mesh = interval_mesh
            else:
                self.mesh = np.concatenate((self.mesh, interval_mesh), axis=0)
        # self.T contains values of the t-parameter which parametrizes the zig-zag path.
        self.T = np.zeros(self.nint*nkp)
        self.points_t = []
        self.points_t_idx = []
        length = 0.0
        for idx in range(self.nint*self.nkp):
            if idx == 0:
                self.T[idx] = 0.0
            else:
                length += lg.norm(self.mesh[idx]-self.mesh[idx-1])
                self.T[idx] = length
            for point in points_list[len(self.points_t):]:
                if np.allclose(self.mesh[idx], point):
                    self.points_t.append(self.T[idx])
                    self.points_t_idx.append(idx)
                    break
        self.points_t = np.array(self.points_t)

    def __len__(self):
        return len(self.T)


class HubbardSystem():
    """ Class to keep all information about particular calculation: its parameters and thermodynamic properties. """

    def __init__(self):
        # List to store names of all attributes storing physical properties of the system.
        self.saved_phys_prop = []
        # List to store names of all attributes storing thermodynamic properties of the system.
        self.saved_therm_prop = []
        # List to store names of all attributes storing calculation parameters.
        self.saved_calc_prop = []

    def get_calculation_parameters(self, fn="solver.ctqmc.in"):
        """ Method to load information about calculation.
        Saved pamaters are:
            mu,
            U,
            beta,
            nffrq,
            nbfrq,
            qx,
            qy,
            q,
            delta.
        """
        with open(fn, 'r') as file:
            for line in file:
                words = line.split()
                # N2N amplitude
                if words != [] and words[0] == "t1":
                    self.tt = float(words[-1])
                    if "tt" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("tt")
                # Chemical potential
                if words != [] and words[0] == "mune":
                    self.mu = float(words[-1])
                    if "mu" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("mu")
                # Coulomb interaction energy
                if words != [] and words[0] == "U":
                    self.U = float(words[-1])
                    if "U" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("U")
                # Inverse temperature
                if words != [] and words[0] == "beta":
                    self.beta = float(words[-1])
                    if "beta" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("beta")
                # Number of fermionic frequencies in full box
                if words != [] and words[0] == "mfreq":
                    self.mfreq = int(words[-1])
                    if "mfreq" not in self.saved_calc_prop:
                        self.saved_calc_prop.append("mfreq")
                # Number of fermionic frequencies for vertex
                if words != [] and words[0] == "nffrq1":
                    self.nffrq = int(words[-1])
                    if "nffrq" not in self.saved_calc_prop:
                        self.saved_calc_prop.append("nffrq")
                # Number of bosonic frequencies for vertex
                if words != [] and words[0] == "nbfrq1":
                    self.nbfrq = int(words[-1])
                    if "nbfrq" not in self.saved_calc_prop:
                        self.saved_calc_prop.append("nbfrq")
                # x-component of order wawe vectror Q
                if words != [] and words[0] == "Q_x":
                    self.qx = float(words[-1])
                    if "qx" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("qx")
                # y-component of order wawe vectror Q
                if words != [] and words[0] == "Q_y":
                    self.qy = float(words[-1])
                    if "qy" not in self.saved_phys_prop:
                        self.saved_phys_prop.append("qy")
        if "qx" in self.saved_phys_prop and "qy" in self.saved_phys_prop:
            # Order wawe vector q
            self.q = np.array((self.qx, self.qy))
            # Relative incommensurability.
            if "delta" not in self.saved_phys_prop:
                self.saved_phys_prop.append('delta')
            self.delta = (np.pi-self.q[0])/np.pi

    def get_thermodynamic_properties(self, fn="solver.nmat999.dat"):
        """ Method to load thermodynamic information about calculation.
        Saved pamaters are:
            nup,
            ndn,
            n,
            x,
            sz.
        """
        with open(fn, 'r') as file:
            for i in range(1):
                line = file.readline()
            # Local spin up occupancy nup
            self.nup = float(file.readline().split()[1])
            # Local spin down occupancy ndn
            self.ndn = float(file.readline().split()[1])
            for i in range(2):
                line = file.readline()
            self.n = float(file.readline().split()[1])
        self.x = 1 - self.n
        self.sz = (self.nup - self.ndn)/2.0
        for value in ("nup", "ndn", "n", "x", "sz"):
            if not value in self.saved_therm_prop:
                self.saved_therm_prop.append(value)

    def __str__(self):
        """ Method to print all saved parameters and properties."""
        message = ""
        for value_key in self.saved_phys_prop:
            message += f"{value_key}: {getattr(self, value_key)}; "
        message += "\n"
        for value_key in self.saved_therm_prop:
            message += f"{value_key}: {getattr(self, value_key):.4f}; "
        message += "\n"
        for value_key in self.saved_calc_prop:
            message += f"{value_key}: {getattr(self, value_key)}; "
        return message[:-1]

    def print_info(self):
        """ Method to print all saved parameters and properties."""
        message = ""
        for value_key in self.saved_phys_prop:
            message += f"{value_key}: {getattr(self, value_key)}; "
        print(message[:-1])
        message = ""
        for value_key in self.saved_therm_prop:
            message += f"{value_key}: {getattr(self, value_key):.4f}; "
        print(message[:-1])
        message = ""
        for value_key in self.saved_calc_prop:
            message += f"{value_key}: {getattr(self, value_key)}; "
        print(message[:-1])


class iQISTCorrelator():
    """ Base class to represent GF or SE of an impurity model. """

    def __init__(self, hs: HubbardSystem):
        # Frequency mesh on imaginary axis with shape (mfreq,), values are purely real and belong to [0,+inf).
        self.im_mesh = (2*np.arange(hs.mfreq)+1)*np.pi/hs.beta
        # Array of data points on imaginary axis with shape (mfreq,nspin).
        self.im_data = np.zeros((hs.mfreq, 2), dtype=complex)
        # Array of relative errors of data points on imaginary axis with shape (mfreq,nspin), purely real.
        self.im_rel_error = np.zeros((hs.mfreq, 2), dtype=float)
        # HubbardSystem object.
        self.hs = hs
        #
        self.was_continued = False

    def load_data(self, fn: str):
        """ Function to load data written in iQIST format from a file. """
        with open(fn, 'r') as file:
            for sigma in range(2):
                for i in range(self.hs.mfreq):
                    line = file.readline()
                    words = line.split()
                    v = complex(float(words[2]), float(words[3]))
                    v_rel_error = abs(
                        complex(float(words[4]), float(words[5])))/abs(v)
                    self.im_data[i, sigma] = v
                    self.im_rel_error[i, sigma] = v_rel_error
                file.readline()
                file.readline()


class SimpleGF(iQISTCorrelator):
    """ Class to represent GF of an impurity model. """

    def load_gf(self, fn="solver.grn.dat"):
        self.load_data(fn)

    def init_continuation(self, re_mesh):
        """ Method to declare attributes related to the continuation problem. """
        # Real frequencies mesh.
        self.re_mesh = re_mesh
        # Number of real frequencies.
        self.re_nfreq = len(re_mesh)
        # Spectral function values array of shape (re_nfreq,2).
        self.spectral_data = np.zeros((self.re_nfreq, 2), dtype=float)
        # List of maxent_solutions that can be used after calculation to check the results.
        self.maxent_solutions = [None, None]

    def an_continue(self, use_nfreq: int, re_mesh: np.ndarray, verbose=False):
        """ Method to perfom analicial continuation of the GF to real axis. """
        # Initialize data structures.
        self.init_continuation(re_mesh)
        # Perfome continuation for both spin values: up and dn.
        for sigma in range(2):
            if verbose:
                print(f"sigma = {sigma}")
            probl = cont.AnalyticContinuationProblem(
                im_axis=self.im_mesh[:use_nfreq+1], re_axis=self.re_mesh, im_data=self.im_data[:use_nfreq+1, sigma], kernel_mode="freq_fermionic")
            # Default model of a spectral function with zero entropy. Choosed to be flat and normalized.
            default_model = np.ones_like(re_mesh)
            default_model /= np.trapz(default_model, re_mesh)
            # Absolute value of im_data errors.
            err = np.abs(self.im_data[:use_nfreq+1, sigma]) * \
                self.im_rel_error[:use_nfreq+1, sigma]
            # Silent (verbose = False) or loud (verbose = True) mode.
            if verbose:
                sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink',
                                     optimizer='newton', stdev=err, model=default_model)
            else:
                with io.capture_output():
                    sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink',
                                         optimizer='newton', stdev=err, model=default_model)
            # Save the results of the calculation.
            self.spectral_data[:, sigma] = sol.A_opt
            self.maxent_solutions[sigma] = sol


class SimpleSE(iQISTCorrelator):
    """ Class to represent SE of an impurity problem. """

    def load_se(self, fn="solver.sgm999.dat"):
        self.load_data(fn)

    def init_continuation(self, re_mesh):
        """ Method to declare attributes related to the continuation problem. """
        # Same as for the SimpleGF class. #
        self.re_mesh = re_mesh
        self.re_nfreq = len(re_mesh)
        self.spectral_data = np.zeros((self.re_nfreq, 2), dtype=float)
        self.maxent_solutions = [None, None]
        # # #
        # Complex-valued array of the continued SE on the real axis after the Kramers-Kroning transformation.
        self.re_data = np.zeros((self.re_nfreq, 2), dtype=complex)

    def an_continue(self, use_nfreq, re_mesh, verbose=False):
        """ Method to perfom analicial continuation of the GF to real axis. """
        # Initialize data structures.
        self.init_continuation(re_mesh)
        # Asymptotics and reduces SE calculation: SE_reduced(w) = (SE(w) - S0)/S1, where SE(w) \approx S0 + S1 / w at w -> +inf.
        se0 = np.array((self.hs.U*self.hs.ndn, self.hs.U*self.hs.nup))
        se1 = np.array((self.hs.U**2*self.hs.ndn*(1-self.hs.ndn),
                       self.hs.U**2*self.hs.nup*(1-self.hs.nup)))
        se_reduced = self.im_data - \
            np.repeat(se0[None, :], self.hs.mfreq, axis=0)
        se_reduced /= np.repeat(se1[None, :], self.hs.mfreq, axis=0)
        for sigma in range(2):
            probl = cont.AnalyticContinuationProblem(
                im_axis=self.im_mesh[:use_nfreq+1], re_axis=self.re_mesh, im_data=se_reduced[:use_nfreq+1, sigma], kernel_mode="freq_fermionic")
            # Default model of a spectral function with zero entropy. Choosed to be flat and normalized.
            default_model = np.ones_like(re_mesh)
            for i, v in enumerate(default_model):
                if re_mesh[i] < 5.0 or re_mesh[i] > 5.0:
                    default_model[i] = 0.0001
            default_model /= np.trapz(default_model, re_mesh)
            # Absolute value of im_data errors.
            err = np.abs(self.im_data) * self.im_rel_error
            err_reduced = (
                err / np.repeat(se1[None, :], self.hs.mfreq, axis=0))[:use_nfreq+1, sigma]
            # Silent (verbose = False) or loud (verbose = True) mode.
            if verbose:
                sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink',
                                     optimizer='newton', stdev=err_reduced, model=default_model)
            else:
                with io.capture_output():
                    sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink',
                                         optimizer='newton', stdev=err_reduced, model=default_model)
            # Calcultion of complex-valued SE with help of build-in Kramers-Kroning transformation.
            se_ana_cont_object = cont.GreensFunction(
                spectrum=sol.A_opt, wgrid=self.re_mesh, kind='fermionic')
            re_data = se_ana_cont_object.kkt()*se1[sigma] + se0[sigma]
            # Results of the calculation.
            self.spectral_data[:, sigma] = sol.A_opt
            self.maxent_solutions[sigma] = sol
            self.re_data[:, sigma] = re_data.copy()
        self.was_continued = True

    def ImAx(self, n: int):
        """ Method to get SE values on the imaginary frequency axis at matsubara frequencies.

        Args:
            n (int): matsubara frequency number. 0 == the first Fermi matsubara frequency.

        Returns:
            ndarray: with a shape (2,) representing two complex values for spin up and spin dn.
        """
        if n >= 0:
            return self.im_data[n]
        else:
            return np.conjugate(self.im_data[-n-1])

    def get_SE_functions(self):
        """ Method used to generate continuous SE function of real frequency as a resulst of an interpolation of descrete data. """
        if self.was_continued:
            se_up = interpolate.interp1d(
                self.re_mesh, self.re_data[:, 0], kind="quadratic")
            se_dn = interpolate.interp1d(
                self.re_mesh, self.re_data[:, 1], kind="quadratic")
            return se_up, se_dn
        else:
            raise RuntimeError


class LatticeGFImAx():
    """ Class representing full lattice Green Functin on imaginary time axis in the local coordinate system.
    """

    def __init__(self, hs: HubbardSystem, se: SimpleSE):
        self.tt = hs.tt
        self.se = se
        # Reload system physical parameters.
        self.q = hs.q
        self.mu = hs.mu
        self.x = hs.x

    def __call__(self, k: np.ndarray, n: int):
        """ Method to get value of the GF.

        Args:
            k (np.ndarray): 2D wave vector.
            n (int): matsubara frequency number. 0 == the first Fermi matsubara frequency.

        Returns:
             ndarray: with a shape (2,) representing two complex values for spin up and spin dn.
        """
        if n >= 0:
            w = +1j*self.se.im_mesh[n]
        else:
            w = -1j*self.se.im_mesh[-n-1]
        se = self.se.ImAx(n)
        # Dispersion relation.

        def e_k(k):
            return -2*1.0*(np.cos(k[0])+np.cos(k[1]))+4*self.tt*np.cos(k[0])*np.cos(k[1])
        # Local GF matrix at k. Filling according to (9) formula from the article.
        grn = np.zeros((2, 2), dtype=complex)
        grn[0, 0] = w + self.mu - se[0] - (e_k(k-self.q/2)+e_k(k+self.q/2))/2
        grn[1, 1] = w + self.mu - se[1] - (e_k(k-self.q/2)+e_k(k+self.q/2))/2
        grn[0, 1] = +(e_k(k-self.q/2) - e_k(k+self.q/2))/2/1j
        grn[1, 0] = -(e_k(k-self.q/2) - e_k(k+self.q/2))/2/1j
        grn = lg.inv(grn)
        return grn


class SpectralFunction():
    """ Class representing spectral function defined at any energy w and at any wawevector k in the global frame. 
        The spectral function is normalized to unity.
    """

    def __init__(self, hs: HubbardSystem, qshift: np.ndarray, se_up: interpolate.interp1d, se_dn: interpolate.interp1d):
        self.tt = hs.tt
        # SE funcitons should be created by SimpleSE object via get_SE_functions() method.
        self.se_up = se_up
        self.se_dn = se_dn
        #  Shift in wavevector that can be performed.
        self.qshift = qshift
        # Reload system physical parameters.
        self.q = hs.q
        self.mu = hs.mu
        self.x = hs.x

    def __call__(self, k: np.ndarray, w):
        # Dispersion relation with shifted wavevector.
        def e_k(k):
            return -2*1.0*(np.cos(k[0]+self.qshift[0])+np.cos(k[1]+self.qshift[1]))+4*self.tt*np.cos(k[0]+self.qshift[0])*np.cos(k[1]+self.qshift[1])
        # Local GF matrices at k+Q and k-Q respectively. Filling according to (9) formula from the article.
        grn_kpq = np.zeros((2, 2), dtype=complex)
        grn_kmq = np.zeros((2, 2), dtype=complex)
        grn_kpq[0, 0] = w + self.mu - self.se_up(w) - (e_k(k)+e_k(k+self.q))/2
        grn_kpq[1, 1] = w + self.mu - self.se_dn(w) - (e_k(k)+e_k(k+self.q))/2
        grn_kpq[0, 1] = +(e_k(k) - e_k(k+self.q))/2/1j
        grn_kpq[1, 0] = -(e_k(k) - e_k(k+self.q))/2/1j
        grn_kmq[0, 0] = w + self.mu - self.se_up(w) - (e_k(k-self.q)+e_k(k))/2
        grn_kmq[1, 1] = w + self.mu - self.se_dn(w) - (e_k(k-self.q)+e_k(k))/2
        grn_kmq[0, 1] = +(e_k(k-self.q) - e_k(k))/2/1j
        grn_kmq[1, 0] = -(e_k(k-self.q) - e_k(k))/2/1j
        grn_kpq = lg.inv(grn_kpq)
        grn_kmq = lg.inv(grn_kmq)
        # GF in global reference frame matrice calculation. Calculation accroding to the (13) formula from the article.
        grn_temp = np.zeros((2, 2), dtype=complex)
        grn_temp[0, 0] = (grn_kpq[0, 0] + grn_kmq[0, 0])
        grn_temp[1, 1] = (grn_kpq[1, 1] + grn_kmq[1, 1])
        grn_temp[0, 1] = (-grn_kpq[0, 1] + grn_kmq[0, 1])*1j
        grn_temp[1, 0] = (+grn_kpq[1, 0] - grn_kmq[1, 0])*1j
        return -np.imag(grn_temp[0, 0]/4+grn_temp[0, 1]/4+grn_temp[1, 0]/4+grn_temp[1, 1]/4)/np.pi


class iQISTResponse():
    """
    Class to represent wave-vector and  imaginary frequency -dependent response-like matrix-valued functions.

    Parameters
    ----------
    hs : HubbardSystem
        HubbardSystem instance which provides nesessary information to create data structures.

    Attributes
    ----------
    hs : HubbardSystem
        Saved HubbardSystem instance used to create iQISTResponse instance.
    im_mesh : ndarray
        Storage for nonnegative bosonic imaginary frequency values (without imaginary identity).
    nkp : int
        Number of positive k-points along each axis (total number of k-points is (2*nkp + 1)^2 ).
    wv_mesh : ndarray
        k-point mesh along each axis.
    im_data : ndarray
        Complex values of the response function. It's shape is (2*nkp+1,2*nkp+1,hs.nbfrq,4,4).
    interpolated : bool
        Flag where information about performed interpolation is stored.
    """

    def __init__(self, hs: HubbardSystem):
        self.im_mesh = 2*np.arange(hs.nbfrq)*np.pi/hs.beta
        self.hs = hs
        self.interpolated = False
        self.nkp = 0
        self.wv_mesh = np.linspace(-np.pi, np.pi, 2*self.nkp+1, endpoint=True)
        self.im_data = np.zeros(
            (2*self.nkp+1, 2*self.nkp+1, self.hs.nbfrq, 4, 4), dtype=complex)

    def load_from_array(self, data: np.ndarray):
        """
        Loads responce function's values from an ndarray and updates k-mesh-dependent structures respectively.

        Parameters
        ----------
            data : ndarray
                Complex values of the response function. It's shape should be (2*nkp+1,2*nkp+1,hs.nbfrq,4,4).
        """
        if data.shape[0] == data.shape[1]:
            self.nkp = (data.shape[0] - 1) // 2
        else:
            raise TypeError("Number of k-points could not be obtained.")
        self.wv_mesh = np.linspace(-np.pi, np.pi, self.nkp*2+1, endpoint=True)
        self.im_data = np.zeros(
            (2*self.nkp+1, 2*self.nkp+1, self.hs.nbfrq, 4, 4), dtype=complex)
        if self.im_data.shape == data.shape:
            self.im_data = data.copy()
        else:
            raise TypeError(
                "Shape of loaded data structure is incompetible with the response function's shape.")

    def __add__(self, other_response):
        if self.im_data.shape != other_response.im_data.shape:
            raise TypeError("Shapes of summands are incompetible.")
        else:
            new_response = iQISTResponse(self.hs)
            new_response.load_from_array(self.im_data + other_response.im_data)
            return new_response

    def __sub__(self, other_response):
        if self.im_data.shape != other_response.im_data.shape:
            raise TypeError("Shapes of summands are incompetible.")
        else:
            new_response = iQISTResponse(self.hs)
            new_response.load_from_array(self.im_data - other_response.im_data)
            return new_response

    def __matmul__(self, other_response):
        """
        Multiplication of iQISTResponse instances
        point-vise with respect to k-points and imaginary freuencies
        but "matrix-like" with respect to spin multiindexes.
        """
        if self.im_data.shape != other_response.im_data.shape:
            raise TypeError("Shapes of summands are incompetible.")
        else:
            new_response = iQISTResponse(self.hs)
            new_response.load_from_array(self.im_data.copy())
            for k in range(self.hs.nbfrq):
                for iqx in range(self.nkp*2+1):
                    for iqy in range(self.nkp*2+1):
                        new_response.im_data[iqx, iqy, k, :, :] = np.matmul(
                            self.im_data[iqx, iqy, k, :, :], other_response.im_data[iqx, iqy, k, :, :])
            return new_response

    def __neg__(self):
        new_response = iQISTResponse(self.hs)
        new_response.load_from_array(-self.im_data.copy())
        return new_response

    def interpolate(self, verbose=False, method="RB", method_settings={"Nx": 5, "Ny": 5}):
        """ Method to create functions which interpolate response function on the Brillouin zone."""
        self.interpolated = True
        # np.empty is used to make a convenient data structure.
        # As RectBivariateSpline handles only real-valued functions, here i create functions both for real and imaginary parts.
        self.component_functions_re = np.empty(
            (self.hs.nbfrq, 4, 4), dtype=object)
        self.component_functions_im = np.empty(
            (self.hs.nbfrq, 4, 4), dtype=object)
        for kfreq in range(self.hs.nbfrq):
            if verbose:
                print("k = ", kfreq)
            for n in range(4):
                for m in range(4):
                    if method == "fourier":
                        approximant_coefficients = FourierApproximation2D(
                            self.wv_mesh, self.wv_mesh, self.im_data[:, :, kfreq, n, m], method_settings["Nx"], method_settings["Ny"]).approximant.coefficients
                        self.component_functions_re[kfreq, n, m] = TrygPoly2DRealPart(
                            approximant_coefficients)
                        self.component_functions_im[kfreq, n, m] = TrygPoly2DImagPart(
                            approximant_coefficients)
                    elif method == "RB":
                        self.component_functions_re[kfreq, n, m] = interpolate.RectBivariateSpline(
                            self.wv_mesh, self.wv_mesh, np.real(self.im_data[:, :, kfreq, n, m]), kx=5, ky=5)
                        self.component_functions_im[kfreq, n, m] = interpolate.RectBivariateSpline(
                            self.wv_mesh, self.wv_mesh, np.imag(self.im_data[:, :, kfreq, n, m]), kx=5, ky=5)
            # Возможно стоит прикрутить вывод ошибки интерполяции для verbose = True.

    def refine_wv_mesh(self, new_nkp: int):
        self.interpolate()
        full_new_nkp = new_nkp*2 + 1
        new_im_data = np.zeros(
            (full_new_nkp, full_new_nkp, self.hs.nbfrq, 4, 4), dtype=complex)
        Kaxis = np.linspace(-np.pi, np.pi, full_new_nkp, endpoint=True)
        for iqx, qx in enumerate(Kaxis):
            for iqy, qy in enumerate(Kaxis):
                for k in range(self.hs.nbfrq):
                    new_im_data[iqx, iqy, k, :, :] = self(v2d(qx, qy), k)
        new_response = iQISTResponse(self.hs)
        new_response.load_from_array(new_im_data)
        return new_response

    def __call__(self, q: np.ndarray, k: int):
        """ After the interpolation precedure the response function can be called as a matrix-valued function of
        k: bosonic imaginary frequency index;
        qx_i and qy_i: wave vector components.
        It returns a matrix of the shape (4,4) ."""
        qx = periodic(q[0])
        qy = periodic(q[1])
        if self.interpolated:
            tmp_matrix = np.zeros((4, 4), dtype=complex)
            for n in range(4):
                for m in range(4):
                    # Collects real and imaginary parts into one complex value.
                    tmp_matrix[n, m] = self.component_functions_re[k][n][m](
                        qx, qy) + 1j*self.component_functions_im[k][n][m](qx, qy)
            return tmp_matrix
        else:
            raise RuntimeError("Interpolation wasn't performed yet.")


class Phi(iQISTResponse):
    """ Class to represent lattice phi function (on the imaginary axis)."""

    def load_from_file(self, nkp: int, fn="nonloc.phi.dat"):
        """ Function to load data written in iQIST format from a nonloc.phi.dat file."""
        im_data = np.zeros(
            (2*nkp+1, 2*nkp+1, self.hs.nbfrq, 4, 4), dtype=complex)
        file = open(fn, 'r')
        for line in file:
            words = line.split()
            if words != [] and words[0] != '#':
                k = int(words[0])-1
                iqx = int(words[1])+nkp
                iqy = int(words[2])+nkp
                n = int(words[3])-1
                m = int(words[4])-1
                re = float(words[5])
                im = float(words[6])
                im_data[iqx, iqy, k, n, m] = complex(re, im)
        file.close()
        self.load_from_array(im_data)

    def get_eigenvalues(self, k: int, qx, qy):
        """ Method to return eigenvalues of the [I - \\phi \\hat{U}] matrix as function of
        k: bosonic imaginary frequency index;
        qx and qy: wave vector components.
        It returns sorted vector of the shape (4,). """
        if self.interpolated:
            U_matrix = np.zeros((4, 4))
            U_matrix[0, 3] = U_matrix[3, 0] = -self.hs.U
            U_matrix[1, 1] = U_matrix[2, 2] = +self.hs.U
            tmp_matrix = np.identity(
                4) - np.matmul(self(v2d(qx, qy), k), U_matrix)
            evs = lg.eigvals(tmp_matrix)
            return np.sort(evs)
        else:
            raise RuntimeError

    def find_min_eigenvalue(self, k=0):
        """ Method to find the wave vector where the smallest eigenvalue becomes minimal.
        returns: min_wv - wave vector, where the smallest eigenvalue becomes minimal;
                 needed_regularization - needed regularization element to remove negative values of the eigenvalues."""
        if self.interpolated:
            minres = opt.minimize(lambda x: np.real(self.get_eigenvalues(k, x[0], x[1])[
                                  0]), vBZ("S")+v2d(0.1*np.pi, 0.0*np.pi), method="Nelder-Mead")
            if minres.success:
                min_wv = minres.x
                needed_regularization = self.get_eigenvalues(
                    k, min_wv[0], min_wv[1])[0]
                if np.imag(needed_regularization) < FLOATZERO:
                    return min_wv, -np.real(needed_regularization)
                else:
                    raise RuntimeError
            else:
                raise RuntimeError(
                    "Optimization did not succeeded."+str(minres))
        else:
            raise RuntimeError

    def compute_chi(self, nkp, regularization=FLOATZERO, verbose="False"):
        """Method to compute the susceptibility chi from phi with regularization and for an updated value of nkp.

        Args:
            nkp (int): 2*nkp+1 will serve as a number of points in wave vector mesh on the Brillouin zone.
            regularization (float, optional): small regulatization parameter used to eliminate numerical singularities and areas of negative susceptibility values. Defaults to FLOATZERO.

        Returns:
            tuple[np.ndarray, np.ndarray]: (new_wv_mesh, chi_mesh),
                where new_wv_mesh has a shape (nkp*2+1,nkp*2+1) and stores wave vector points on the Brillouin zone,
                and chi_mesh has a shape (nbfrq,2*nkp+1,2*nkp+1,nidx,nidx) and stores calculated chi complex values.
        """
        if regularization < 0.0:
            regularization = FLOATZERO
        if self.interpolated:
            U_matrix = np.zeros((4, 4))
            U_matrix[0, 3] = U_matrix[3, 0] = -self.hs.U
            U_matrix[1, 1] = U_matrix[2, 2] = +self.hs.U
            new_wv_mesh = np.linspace(-np.pi, np.pi, 2*nkp+1, endpoint=True)
            chi_mesh = np.zeros(
                (2*nkp+1, 2*nkp+1, self.hs.nbfrq, 4, 4), dtype=complex)
            for k in range(self.hs.nbfrq):
                if verbose:
                    print("k = ", k)
                for iqx in range(2*nkp+1):
                    for iqy in range(2*nkp+1):
                        qx = new_wv_mesh[iqx]
                        qy = new_wv_mesh[iqy]
                        chi_mesh[iqx, iqy, k, :, :] = np.matmul(lg.inv(np.identity(4)*(1+regularization)
                                                                       - np.matmul(self(v2d(qx, qy), k), U_matrix)), self(v2d(qx, qy), k))
            return new_wv_mesh, chi_mesh
        else:
            raise RuntimeError


class SingularPart(iQISTResponse):
    def __init__(self, hs: HubbardSystem):
        super().__init__(hs)
        self.U_matrix = np.zeros((4, 4))
        self.U_matrix[0, 3] = self.U_matrix[3, 0] = -self.hs.U
        self.U_matrix[1, 1] = self.U_matrix[2, 2] = +self.hs.U

    def compute_from_phi(self, phi: iQISTResponse):
        raise NotImplementedError

    def get_eigenvalues(self, q: np.ndarray, k: int):
        """ Method to return eigenvalues of the self matrix as function of
        k: bosonic imaginary frequency index;
        q: wave vector.
        It returns sorted vector of the shape (4,). """
        if self.interpolated:
            tmp_matrix = self(q, k)
            evs = lg.eigvals(tmp_matrix)
            return np.sort(evs)
        else:
            raise RuntimeError

    def get_mineigenvalue(self, q: np.ndarray, k: int):
        return np.real(self.get_eigenvalues(q, k)[0])

    def find_local_min_eigenvalue(self, k: int, initial_value: np.ndarray):
        """ Method to find the wave vector where the smallest eigenvalue becomes minimal.
        returns: min_wv - wave vector, where the smallest eigenvalue becomes minimal;
                 min_eigen_value - local minimal value of real smallest eigenvalue."""
        if self.interpolated:
            minres = opt.minimize(lambda q: self.get_mineigenvalue(
                q, k), initial_value, method="Nelder-Mead")
            if minres.success:
                min_wv = minres.x
                min_eigen_value = self.get_eigenvalues(min_wv, k)[0]
                if np.imag(min_eigen_value) < FLOATZERO:
                    return min_wv, np.real(min_eigen_value)
                else:
                    raise RuntimeError(
                        "Imaginary part of the smallest eigen value is not zero.")
            else:
                raise RuntimeError(
                    "Optimization did not succeeded."+str(minres))
        else:
            raise RuntimeError("Interpolation was not performed.")

    def get_regularized_inverse(self, regularization):
        inverse_data = self.im_data + np.identity(4)*regularization
        for k in range(self.hs.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    inverse_data[iqx, iqy, k, :, :] = lg.inv(
                        inverse_data[iqx, iqy, k, :, :])
        regularizaed_inverse = iQISTResponse(self.hs)
        regularizaed_inverse.load_from_array(inverse_data)
        self.regularization = regularization
        return regularizaed_inverse


class SingularPartLeft(SingularPart):
    def compute_from_phi(self, phi: iQISTResponse):
        self.nkp = phi.nkp
        self.wv_mesh = phi.wv_mesh.copy()
        self.im_data = phi.im_data.copy()
        for k in range(self.hs.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    self.im_data[iqx, iqy, k, :, :] = np.identity(
                        4) - np.matmul(self.im_data[iqx, iqy, k, :, :], self.U_matrix)


class SingularPartRight(SingularPart):
    def compute_from_phi(self, phi: iQISTResponse):
        self.nkp = phi.nkp
        self.wv_mesh = phi.wv_mesh.copy()
        self.im_data = phi.im_data.copy()
        for k in range(self.hs.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    self.im_data[iqx, iqy, k, :, :] = np.identity(
                        4) - np.matmul(self.U_matrix, self.im_data[iqx, iqy, k, :, :])


class iQISTResponseSpinRepr(iQISTResponse):
    def __init__(self, respons: iQISTResponse):
        super().__init__(respons.hs)
        self.load_from_array(respons.im_data)
        # Just the definition of Pauli matrices multiplied by factor 2.
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        self.Sigma = [sigma_x, sigma_y, sigma_z]
        # Matrices to pass from default basis to spin basis.
        self.TR = np.zeros((4, 3), dtype=complex)
        self.TL = np.zeros((3, 4), dtype=complex)
        for m in range(3):
            for sigma1 in range(2):
                for sigma2 in range(2):
                    alpha = 2*sigma1+sigma2
                    self.TR[alpha, m] = self.Sigma[m][sigma1, sigma2]
                    self.TL[m, alpha] = self.Sigma[m][sigma2, sigma1]
        self.spin_im_data = np.zeros(
            self.im_data.shape[:-2]+(3, 3), dtype=complex)
        for k in range(self.hs.nbfrq):
            for iqx in range(self.nkp*2+1):
                for iqy in range(self.nkp*2+1):
                    self.spin_im_data[iqx, iqy, k, :,
                                      :] = self.TL @ self.im_data[iqx, iqy, k, :, :] @ self.TR

    def __call__(self, q: np.ndarray, k: int):
        result = super().__call__(q, k)
        result = self.TL @ result @ self.TR
        return result


class Chi(iQISTResponse):
    """
    Class to represent lattice susceptibility chi function, including its analytic continuation to the real frequency axis.

    Attributes
    ----------
    TR : ndarray
        Right matrix for transofrmation to spin representation.
    TL : ndarray
        Left matrix for transformation to spin representation.
    Sigma : list
        List of 4 Pauli matrices devided by factor of 2.
    continued : bool
        Flag which stores information if the analytic continuation was performed.
    im_data_spin : ndarray
        Complex values of the response function in spin representation. It's shape is (2*nkp+1,2*nkp+1,hs.nbfrq,3,3).
    """

    def __init__(self, hs: HubbardSystem):
        super().__init__(hs)
        sigma_x = np.array([[0.0, 1.0], [1.0,  0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1j], [1j,  0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        sigma_0 = np.array([[1.0, 0.0], [0.0,  1.0]], dtype=complex)
        self.Sigma = [sigma_x, sigma_y, sigma_z, sigma_0]
        self.continued = False
        # Matrices to pass from default basis to spin basis.
        self.TR = np.zeros((4, 4), dtype=complex)
        self.TL = np.zeros((4, 4), dtype=complex)
        for m in range(4):
            for sigma1 in range(2):
                for sigma2 in range(2):
                    alpha = 2*sigma1+sigma2
                    self.TL[m, alpha] = self.Sigma[m][sigma1, sigma2]/2.0
                    self.TR[alpha, m] = self.Sigma[m][sigma2, sigma1]/2.0

    def get_spin_data(self) -> np.ndarray:
        temp = self.im_data.copy()
        shape = list(temp.shape)
        # shape[-2] = 4
        # shape[-1] = 4
        temp_spin = np.zeros(shape, dtype=complex)
        for iqx in range(self.nkp*2+1):
            for iqy in range(self.nkp*2+1):
                for k in range(self.hs.nbfrq):
                    temp_spin[iqx, iqy, k, :,
                              :] = self.TL @ temp[iqx, iqy, k, :, :] @ self.TR
        self.im_data_spin = temp_spin

    def load_from_array(self, data: np.ndarray):
        super().load_from_array(data)
        self.get_spin_data()

    def __add__(self, other_response):
        temp_data = super().__add__(other_response).im_data
        temp_chi = Chi(self.hs)
        temp_chi.load_from_array(temp_data)
        return temp_chi

    def __matmul__(self, other_response):
        temp_data = super().__matmul__(other_response).im_data
        temp_chi = Chi(self.hs)
        temp_chi.load_from_array(temp_data)
        return temp_chi

    def __call__(self, k: int, qx_i, qy_i, representation="spin"):
        # Application of periodicity condition.
        qx = periodic(qx_i)
        qy = periodic(qy_i)
        if self.interpolated:
            tmp_matrix_default = np.zeros((4, 4), dtype=complex)
            for n in range(4):
                for m in range(4):
                    # Collects real and imaginary parts into one complex value.
                    tmp_matrix_default[n, m] = self.component_functions_re[k][n][m](
                        qx, qy) + 1j*self.component_functions_im[k][n][m](qx, qy)
            if representation == "default":
                return tmp_matrix_default
            else:
                tmp_matrix_spin = np.matmul(
                    np.matmul(self.TL, tmp_matrix_default), self.TR)
                if representation == "spin":
                    return tmp_matrix_spin
                elif representation == "rotational":
                    # To be done in the future if is needed.
                    raise NotImplementedError(
                        "Rotational representation is not implemented yet.")
                else:
                    raise TypeError(
                        "Unrecognized susceptibility's representation type.")
        else:
            raise RuntimeError("Interpolation was not performed correctly.")

    def initialize_continuation(self, wv_path: BZPath, re_mesh: np.ndarray, component: tuple):
        """Method to prepare susceptibility object for analytic continuation and to load appropriate data structures.

        Args:
            wv_path (BZPath): the path in the Brillouin zone along which the spectral density will be calculated.
            re_mesh (np.ndarray): positive real frequency mesh.
            component (tuple): tuple of a shape (s1,s2), where s1 and s2 are strings from {'x','y','z'} set.
        """
        self.continuation_wv_path = wv_path
        self.continuation_re_mesh = re_mesh
        self.continuation_re_data = np.zeros(
            (len(self.continuation_wv_path), len(re_mesh)), dtype=float)
        if len(component) == 2 and component[0] in ('x', 'y', 'z') and component[1] in ('x', 'y', 'z'):
            self.continuation_representation = "spin"
            self.continuation_component = (spin2index(
                component[0]), spin2index(component[1]))
        else:
            raise NotImplementedError(
                "Only spin components of the susceptibility can be continued in that version.")

    def an_continue(self, model_function=None):
        """Method to get the spectral function corresponding to the susceptibility component via analytic continuation.

        Args:
            model_function (function, optional): default spectral density function, which serves as zero for the entropy function in Maximum Entropy method.
                                                Defaults to None (сonstant spectral density).
        """
        for iq, q in enumerate(self.continuation_wv_path.mesh):
            print(f"{iq + 1} from {len(self.continuation_wv_path.mesh)}")
            im_data = np.zeros(self.hs.nbfrq, dtype=float)
            for k in range(self.hs.nbfrq):
                im_data[k] = np.real(self(k, *q, representation="spin")
                                     [self.continuation_component[0], self.continuation_component[1]])
            probl = cont.AnalyticContinuationProblem(im_axis=self.im_mesh, re_axis=self.continuation_re_mesh,
                                                     im_data=im_data, kernel_mode='freq_bosonic')
            if model_function is None:
                model = np.ones_like(self.continuation_re_mesh)
            else:
                model = model_function(self.continuation_re_mesh)
            model /= np.trapz(model, self.continuation_re_mesh)
            # Default value of error. May be should be passed as an argument.
            err = im_data * 0.001
            try:
                with io.capture_output():
                    solution, _ = probl.solve(
                        method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=model)
                    self.continuation_re_data[iq,
                                              :] = solution.A_opt * self.continuation_re_mesh
            except:
                print("Error at :", iq)
                self.continuation_re_data[iq, :] = 0.0
        self.continued = True

    def get_dispersion(self, max_w=np.inf):
        """Method to get the magnon dispersion curve after the analytic continuation.

        Args:
            max_w (float, optional): Spectral maximum is searched on the interval [0,max_w]. Defaults to np.inf.

        Raises:
            RuntimeError: is the analytic continuation was not yet performed.

        Returns:
            tuple: dispersion_T, dispersion_w
                both are numpy ndarrays of the shape (BZPath.nint,BZPath.nkp) .
                dispersion_T stores BZPath.nint*BZPath.nkp values of path parametrization parameter t .
                dispersion_w stores BZPath.nint*BZPath.nkp values of corresponding energy maxima.
        """
        if self.continued:
            wv_path = self.continuation_wv_path
            nint = wv_path.nint
            nkp = wv_path.nkp
            dispersions_T = np.zeros((nint, nkp))
            dispersions_w = np.zeros((nint, nkp))
            for pint_idx in range(nint):
                for iq in range(nkp):
                    w_mesh = self.continuation_re_mesh
                    chi_mesh = self.continuation_re_data[pint_idx*nkp+iq, :]
                    w_max = w_mesh[np.argmax(chi_mesh[w_mesh < max_w])]
                    dispersions_T[pint_idx, iq] = wv_path.T[pint_idx*nkp+iq]
                    dispersions_w[pint_idx, iq] = w_max
            return dispersions_T, dispersions_w
        else:
            raise RuntimeError(
                "Analytic continuation was not yet performed.")


class Calculation():
    def __init__(self, h5fn, bfsym=False):
        self.__get_params(h5fn)

        system = HubbardSystem()
        system.nkp = self.params["nkp"]
        system.beta = self.params["beta"]
        system.U = self.params["U"]
        self.hsystem = system
        if bfsym:
            system.nbfrq = 2*self.params["nbfrq"]-1
            self.bfreqs = (np.arange(
                2*self.params["nbfrq"]-1) - self.params["nbfrq"] + 1)*2*np.pi/self.params["beta"]
        else:
            system.nbfrq = self.params["nbfrq"]
            self.bfreqs = np.arange(
                self.params["nbfrq"])*2*np.pi/self.params["beta"]

        self.chi0 = Chi(system)
        with h5py.File(h5fn, 'r') as f:
            chi0_data = np.array(f["chi0_closed_real"]).T + \
                np.array(f["chi0_closed_imag"]).T*1j
        self.chi0.load_from_array(chi0_data)

        self.inv_chi0 = Chi(system)
        inv_chi0_data = np.empty_like(self.chi0.im_data)
        nkp_x, nkp_y, full_nbfrq = self.chi0.im_data.shape[:3]
        for iqx in range(nkp_x):
            for iqy in range(nkp_y):
                for k in range(full_nbfrq):
                    inv_chi0_data[iqx, iqy, k, :, :] = scipy.linalg.inv(
                        self.chi0.im_data[iqx, iqy, k, :, :])
        self.inv_chi0.load_from_array(inv_chi0_data)

        with h5py.File(h5fn, 'r') as f:
            phi_data = np.array(f["phi_real"]).T + np.array(f["phi_imag"]).T*1j

        self.phi = Chi(system)
        self.phi.load_from_array(phi_data)

        singular_part_right = SingularPartRight(system)
        singular_part_right.compute_from_phi(self.phi)
        inverse_singular_part_right = singular_part_right.get_regularized_inverse(
            0.0)

        self.chi = Chi(system)
        self.chi.load_from_array(
            (self.phi @ inverse_singular_part_right).im_data)

        self.inv_chi = Chi(system)
        inv_chi_data = np.empty_like(self.chi.im_data)
        nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
        for iqx in range(nkp_x):
            for iqy in range(nkp_y):
                for k in range(full_nbfrq):
                    inv_chi_data[iqx, iqy, k, :, :] = scipy.linalg.inv(
                        self.chi.im_data[iqx, iqy, k, :, :])
        self.inv_chi.load_from_array(inv_chi_data)

        self.U_matrix = np.zeros((4, 4))
        self.U_matrix[0, -1] = -self.params["U"]
        self.U_matrix[-1, 0] = -self.params["U"]
        self.U_matrix[1, 1] = self.params["U"]
        self.U_matrix[2, 2] = self.params["U"]

        self.extended_U_matrix = Chi(system)
        extended_U_matrix_data = np.empty_like(self.chi.im_data)
        nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
        for iqx in range(nkp_x):
            for iqy in range(nkp_y):
                for k in range(full_nbfrq):
                    extended_U_matrix_data[iqx, iqy, k, :, :] = self.U_matrix
        self.extended_U_matrix.load_from_array(extended_U_matrix_data)

    def __get_params(self, h5fn):
        params = {}
        with h5py.File(h5fn, 'r') as f:
            attrs = f.attrs
            nffrq = attrs["nffrq"][0]
            nbfrq = attrs["nbfrq"][0]
            beta = attrs["beta"][0]
            U = attrs["U"][0]
            nup = attrs["n_up"][0]
            ndn = attrs["n_dn"][0]
            sz = (nup - ndn)/2.0
            chi0_data = np.array(f["chi0_closed_real"]).T + \
                np.array(f["chi0_closed_imag"]).T*1j
            nkp = (chi0_data.shape[0]-1) // 2
        params["nffrq"] = nffrq
        params["nbfrq"] = nbfrq
        params["beta"] = beta
        params["U"] = U
        params["nup"] = nup
        params["ndn"] = ndn
        params["sz"] = sz
        params["nkp"] = nkp
        self.params = params


if __name__ == "__main__":
    print("Imports fine.")
