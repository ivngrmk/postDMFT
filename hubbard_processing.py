import numpy as np
import ana_cont.continuation as cont
from triqs.gf import GfImFreq
from IPython.utils import io

class HubbardSystem():
    """ Class to keep all information about particular calculation: its parameters and thermodynamic properties. """
    def __init__(self,path='./',verbose=False):
        self.saved_values = []
        if verbose:
            print(f"Q = {self.Q}, delta_Q = {self.delta}")
            print(f"mu = {self.mu:.3f}, n = {self.n:.5f}, x = {self.x:.5f}, n_up = {self.n_up}, n_dn = {self.n_dn}, U = {self.U}, sz = {0.5*(self.n_up-self.n_dn)}")
        
    def get_calculation_parameters(self,fn = "solver.ctqmc.dat"):
        """ Method to load information about calculation.
        Saved pamaters are:
            mu,
            U,
            beta,
            nffrq,
            nbfrq,
            qx,
            qy,
            Q,
            delta.
        """
        with open(fn,'r') as file:
            for line in file:
                words = line.split()
                # Chemical potential
                if words != [] and words[0] == "mune":
                    self.mu = float(words[-1])
                    if "mune" not in self.saved_values:
                        self.saved_values.append("mune")
                # Coulomb interaction energy
                if words != [] and words[0] == "U":
                    self.U = float(words[-1])
                    if "U" not in self.saved_values:
                        self.saved_values.append("U")
                # Inverse temperature
                if words != [] and words[0] == "beta":
                    self.beta = float(words[-1])
                    if "beta" not in self.saved_values:
                        self.saved_values.append("beta")
                # Number of fermionic frequencies
                if words != [] and words[0] == "nffrq1":
                    self.nffrq = int(words[-1])
                    if "nffrq" not in self.saved_values:
                        self.saved_values.append("nffrq")
                # Number of bosonic frequencies
                if words != [] and words[0] == "nbfrq1":
                    self.nbfrq = int(words[-1])
                    if "nbfrq" not in self.saved_values:
                        self.saved_values.append("nbfrq")
                # x-component of order wawe vectror Q
                if words != [] and words[0] == "Q_x":
                    self.qx = float(words[-1])
                    if "qx" not in self.saved_values:
                        self.saved_values.append("qx")
                # y-component of order wawe vectror Q    
                if words != [] and words[0] == "Q_y":
                    self.qy = float(words[-1])
                    if "qy" not in self.saved_values:
                        self.saved_values.append("qy")
        # Order wawe vector Q
        self.Q = np.array((self.Qx,self.Qy))
        # Relative incommensurability.
        self.delta = (np.pi-self.Q[0])/np.pi

    def get_thermodynamic_properties(self,fn = "solver.nmat999.dat"):
        """ Method to load thermodynamic information about calculation.
        Saved pamaters are:
            nup,
            ndn,
            n,
            x,
            sz.
        """
        with open(fn,'r') as file:
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
        for value in ("nup","ndn","n","x","sz"):
            if not value in self.saved_values:
                self.saved_values.append(value)

    def print_info(self):
        """ Method to print all saved parameters and properties."""
        message = ""
        for value_key in self.saved_values:
            message += f"{value_key}: {getattr(self,value_key):.4f}; "
        print(message[:-1])

def an_continue(kernel_mode, im_freqs, im_data, relative_errors, re_freqs, verbose=False):
    """ Function to perform analytic continuation from imaginary axis to real axis.
    Arguments list:
        kernel_mode: "fermionic" or "bosonic",
        im_data: complex (for fermionic) or purely real (for bosonic) values of continuated quantity on imaginary axis with (n_im_freqs,) shape,
        im_freqs: absolute values of frequencies on imaginary axis from [0,+inf) with (n_im_freqs,) shape,
        re_freqs: values of frequencies on real axis with (n_re_freqs,) shape,
        relative_errors: relative errors of absolute values of im_data with (n_im_freqs,) shape,
        verbose: True or False, if True then full continuation output will be printed.
    Returns: Imaginary part of continuated quantity on real axis with (n_re_freqs,) shape.
    """
    # Input arguments check.
    if not kernel_mode in ("bosonic", "fermionic"):
        raise ValueError("Kernel_mode must be 'bosonic' or 'fermionic'")
    if not (np.isreal(im_freqs) and np.isreal(re_freqs) and np.isreal(relative_errors)):
        raise ValueError("Frequencies and errors should be purely real")
    if kernel_mode == "bosonic":
        if not np.isreal(im_data):
            raise ValueError("In bosonic case data on imaginaty axis should be purely real.")
    probl = cont.AnalyticContinuationProblem(im_axis=im_freqs, re_axis=re_freqs, im_data=im_data, kernel_mode=kernel_mode)
    # Default model with zero entropy. Choosed to be flat and normalized.
    default_model = np.ones_like(re_freqs)
    default_model /= np.trapz(default_model, re_freqs)
    # Absolute value of im_data errors.
    err = np.abs(im_data) * relative_errors
    # Silent (verbose = False) or loud (verbose = True) mode.
    if verbose:
        sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=default_model)
    else:
        with io.capture_output():
            sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=default_model)
    re_data = sol.A_opt
    # In a bosonic case ana_cont returns S(w) = Im(G(w))/w, so multiplication on frequency w should be performed.
    if kernel_mode == "bosonic":
        re_data = re_data*re_freqs
    # Output:
    if kernel_mode == "fermionic":
        return re_data, sol.backtransform.real+1j*sol.backtransform.imag
    if kernel_mode == "bosonic":
        return re_data, sol.backtransform.real

"""
class iQISTGfImFreq(GfImFreq):
    def get_data_from_iqist_file(self, fullfn):
        data = np.loadtxt(fullfn)
        up_data = [1/1j/v[1] for v in data if v[0] == 1][:self.mesh.last_index()+1]
        dn_data = [1/1j/v[1] for v in data if v[0] == 2][:self.mesh.last_index()+1]
        up_data = [complex(v[2],v[3]) for v in data if v[0] == 1][:self.mesh.last_index()+1]
        dn_data = [complex(v[2],v[3]) for v in data if v[0] == 2][:self.mesh.last_index()+1]
        up_data = list(np.conj(np.array(up_data)))[::-1] + up_data
        dn_data = list(np.conj(np.array(dn_data)))[::-1] + dn_data
        self.data[:,0,0] = np.array(up_data)
        self.data[:,1,1] = np.array(dn_data)

    def compute_tail_fit(self,expansion_order,nmin,nmax,known_moments=None):
        beta = self.mesh.beta
        if known_moments is None:
            shape = [0] + list(self.target_shape)
            known_moments = np.zeros(shape, dtype=complex)
        nmin = 20
        o_min = (2*nmin+1)*np.pi/beta
        nmax = 60
        o_max = (2*nmax+1)*np.pi/beta
        tail, err = self.fit_hermitian_tail_on_window(n_min = nmin, 
                                                      n_max = nmax , 
                                                      known_moments = known_moments, 
                                                      n_tail_max = 10 * len(self.mesh) , 
                                                      expansion_order = expansion_order)
        self.tail = tail
        return tail, err

    def compute_tail(self,w,m,n,expansion_order=None):
        tail = self.tail[:,m,n]
        v = 0.0
        if not expansion_order or expansion_order > len(tail):
            expansion_order = len(tail)
        for n in range(expansion_order):
            v += tail[n]/w**n
        return v

    def ana_cont(self,im_np,re_freqs,rerror=0.001,verbose=False):
        im_freqs = np.zeros(im_np)
        im_data  = np.zeros([im_np]+list(self.target_shape),dtype=complex)
        for i in range(im_np):
            for m in range(self.target_shape[0]):
                for n in range(self.target_shape[1]):
                    im_data[i,m,n] = self.data[im_np + i,m,n]
            im_freqs[i] = np.imag(self.mesh(i))
        re_data = np.zeros([len(re_freqs)]+list(self.target_shape))
        for m in range(self.target_shape[0]):
            for n in range(self.target_shape[1]):
                if m == n:
                    re_data[:,m,n] = an_continue_fermionic(im_data[:,m,n], im_freqs, re_freqs, rerror=rerror, verbose=verbose)
        return re_data
"""

if __name__ == "__main__":
    print("Imports fine.")
