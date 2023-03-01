import numpy as np
import ana_cont.continuation as cont
# from triqs.gf import GfImFreq
from IPython.utils import io
from scipy import interpolate
from scipy import linalg as lg

# Details of ana_cont API and usage can be find: https://josefkaufmann.github.io/ana_cont/api_doc.html and https://arxiv.org/abs/2105.11211 .

def v2d(vx,vy):
    """ Functions to easily create 2d wave vectors. """
    v = np.zeros(2)
    v[0] = vx; v[1] = vy
    return v

class HubbardSystem():
    """ Class to keep all information about particular calculation: its parameters and thermodynamic properties. """
    def __init__(self):
        # List to store names of all attributes storing physical properties of the system.
        self.saved_phys_prop = []
        # List to store names of all attributes storing thermodynamic properties of the system.
        self.saved_therm_prop = []
        # List to store names of all attributes storing calculation parameters.
        self.saved_calc_prop = []
        
        
    def get_calculation_parameters(self,fn = "solver.ctqmc.in"):
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
        with open(fn,'r') as file:
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
        # Order wawe vector q
        self.q = np.array((self.qx,self.qy))
        # Relative incommensurability.
        if "delta" not in self.saved_phys_prop:
            self.saved_phys_prop.append('delta')
        self.delta = (np.pi-self.q[0])/np.pi

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
            if not value in self.saved_therm_prop:
                self.saved_therm_prop.append(value)

    def print_info(self):
        """ Method to print all saved parameters and properties."""
        message = ""
        for value_key in self.saved_phys_prop:
            message += f"{value_key}: {getattr(self,value_key)}; "
        print(message[:-1])
        message = ""
        for value_key in self.saved_therm_prop:
            message += f"{value_key}: {getattr(self,value_key):.4f}; "
        print(message[:-1])
        message = ""
        for value_key in self.saved_calc_prop:
            message += f"{value_key}: {getattr(self,value_key)}; "
        print(message[:-1])

class iQISTCorrelator():
    """ Base class to represent GF or SE of an impurity model. """
    def __init__(self, hs: HubbardSystem):
        # Frequency mesh on imaginary axis with shape (mfreq,), values are purely real and belong to [0,+inf).
        self.im_mesh = (2*np.arange(hs.mfreq)+1)*np.pi/hs.beta
        # Array of data points on imaginary axis with shape (mfreq,nspin).
        self.im_data = np.zeros((hs.mfreq,2),dtype=complex)
        # Array of relative errors of data points on imaginary axis with shape (mfreq,nspin), purely real.
        self.im_rel_error = np.zeros((hs.mfreq,2),dtype=float)
        # HubbardSystem object.
        self.hs = hs
        #
        self.was_continued = False

    def load_data(self,fn: str):
        """ Function to load data written in iQIST format from a file. """
        with open(fn,'r') as file:
            for sigma in range(2):
                for i in range(self.hs.mfreq):
                    line = file.readline()
                    words = line.split()
                    v = complex(float(words[2]),float(words[3]))
                    v_rel_error = abs(complex(float(words[4]),float(words[5])))/abs(v)
                    self.im_data[i,sigma] = v
                    self.im_rel_error[i,sigma] = v_rel_error
                file.readline()
                file.readline()


class SimpleGF(iQISTCorrelator):
    """ Class to represent GF of an impurity model. """
    def load_gf(self,fn="solver.grn.dat"):
        self.load_data(fn)

    def init_continuation(self,re_mesh):
        """ Method to declare attributes related to the continuation problem. """
        # Real frequencies mesh.
        self.re_mesh = re_mesh
        # Number of real frequencies.
        self.re_nfreq = len(re_mesh)
        # Spectral function values array of shape (re_nfreq,2). 
        self.spectral_data = np.zeros((self.re_nfreq,2),dtype=float)
        # List of maxent_solutions that can be used after calculation to check the results.
        self.maxent_solutions = [None,None]

    def an_continue(self,use_nfreq: int,re_mesh: np.ndarray,verbose=False):
        """ Method to perfom analicial continuation of the GF to real axis. """
        # Initialize data structures.
        self.init_continuation(re_mesh)
        # Perfome continuation for both spin values: up and dn.
        for sigma in range(2):
            if verbose:
                print(f"sigma = {sigma}")
            probl = cont.AnalyticContinuationProblem(im_axis=self.im_mesh[:use_nfreq+1], re_axis=self.re_mesh, im_data=self.im_data[:use_nfreq+1,sigma], kernel_mode="freq_fermionic")
            # Default model of a spectral function with zero entropy. Choosed to be flat and normalized.
            default_model = np.ones_like(re_mesh)
            default_model /= np.trapz(default_model, re_mesh)
            # Absolute value of im_data errors.
            err = np.abs(self.im_data[:use_nfreq+1,sigma]) * self.im_rel_error[:use_nfreq+1,sigma]
            # Silent (verbose = False) or loud (verbose = True) mode.
            if verbose:
                sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=default_model)
            else:
                with io.capture_output():
                    sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=default_model)
            # Save the results of the calculation.
            self.spectral_data[:,sigma] = sol.A_opt
            self.maxent_solutions[sigma] = sol
    
class SimpleSE(iQISTCorrelator):
    """ Class to represent SE of an impurity problem. """
    def load_se(self,fn="solver.sgm999.dat"):
        self.load_data(fn)

    def init_continuation(self,re_mesh):
        """ Method to declare attributes related to the continuation problem. """
        # Same as for the SimpleGF class. #
        self.re_mesh = re_mesh
        self.re_nfreq = len(re_mesh)
        self.spectral_data = np.zeros((self.re_nfreq,2),dtype=float)
        self.maxent_solutions = [None,None]
        # # #
        # Complex-valued array of the continued SE on the real axis after the Kramers-Kroning transformation.
        self.re_data = np.zeros((self.re_nfreq,2),dtype=complex)

    def an_continue(self,use_nfreq,re_mesh,verbose=False):
        """ Method to perfom analicial continuation of the GF to real axis. """
        # Initialize data structures.
        self.init_continuation(re_mesh)
        # Asymptotics and reduces SE calculation: SE_reduced(w) = (SE(w) - S0)/S1, where SE(w) \approx S0 + S1 / w at w -> +inf.
        se0 = np.array((self.hs.U*self.hs.ndn,self.hs.U*self.hs.nup))
        se1 = np.array((self.hs.U**2*self.hs.ndn*(1-self.hs.ndn),self.hs.U**2*self.hs.nup*(1-self.hs.nup)))
        se_reduced = self.im_data - np.repeat(se0[None,:],self.hs.mfreq,axis=0)
        se_reduced /= np.repeat(se1[None,:],self.hs.mfreq,axis=0)
        for sigma in range(2):
            probl = cont.AnalyticContinuationProblem(im_axis=self.im_mesh[:use_nfreq+1], re_axis=self.re_mesh, im_data=se_reduced[:use_nfreq+1,sigma], kernel_mode="freq_fermionic")
            # Default model of a spectral function with zero entropy. Choosed to be flat and normalized.
            default_model = np.ones_like(re_mesh)
            for i,v in enumerate(default_model):
                if re_mesh[i] < 5.0 or re_mesh[i] > 5.0:
                    default_model[i] = 0.0001
            default_model /= np.trapz(default_model, re_mesh)
            # Absolute value of im_data errors.
            err = np.abs(self.im_data) * self.im_rel_error
            err_reduced = (err / np.repeat(se1[None,:],self.hs.mfreq,axis=0))[:use_nfreq+1,sigma]
            # Silent (verbose = False) or loud (verbose = True) mode.
            if verbose:
                sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err_reduced, model=default_model)
            else:
                with io.capture_output():
                    sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err_reduced, model=default_model)
            # Calcultion of complex-valued SE with help of build-in Kramers-Kroning transformation.
            se_ana_cont_object = cont.GreensFunction(spectrum = sol.A_opt, wgrid = self.re_mesh, kind = 'fermionic')
            re_data = se_ana_cont_object.kkt()*se1[sigma] + se0[sigma]
            # Results of the calculation.
            self.spectral_data[:,sigma] = sol.A_opt
            self.maxent_solutions[sigma] = sol
            self.re_data[:,sigma] = re_data.copy()
        self.was_continued = True

    def get_SE_functions(self):
        """ Method used to generate continuous SE function of real frequency as a resulst of an interpolation of descrete data. """
        if self.was_continued:
            se_up = interpolate.interp1d(self.re_mesh, self.re_data[:,0],kind="quadratic")
            se_dn = interpolate.interp1d(self.re_mesh, self.re_data[:,1],kind="quadratic")
            return se_up, se_dn
        else:
            raise RuntimeError

class SpectralFunction():
    """ Class representing spectral function defined at any energy w and at any wawevector k. """
    def __init__(self,hs: HubbardSystem, qshift: np.ndarray, se_up: interpolate.interp1d, se_dn: interpolate.interp1d):
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
    
    def __call__(self,k: np.ndarray, w):
        # Dispersion relation with shifted wavevector.
        def e_k(k):
            return -2*1.0*(np.cos(k[0]+self.qshift[0])+np.cos(k[1]+self.qshift[1]))+4*self.tt*np.cos(k[0]+self.qshift[0])*np.cos(k[1]+self.qshift[1])
        # Local GF matrices at k+Q and k-Q respectively. Filling according to (9) formula from the article.
        grn_kpq = np.zeros((2,2),dtype=complex)
        grn_kmq = np.zeros((2,2),dtype=complex)
        grn_kpq[0,0] = w + self.mu - self.se_up(w) - (e_k(k)+e_k(k+self.q))/2
        grn_kpq[1,1] = w + self.mu - self.se_dn(w) - (e_k(k)+e_k(k+self.q))/2
        grn_kpq[0,1] = +(e_k(k) - e_k(k+self.q))/2/1j
        grn_kpq[1,0] = -(e_k(k) - e_k(k+self.q))/2/1j
        grn_kmq[0,0] = w + self.mu - self.se_up(w) - (e_k(k-self.q)+e_k(k))/2
        grn_kmq[1,1] = w + self.mu - self.se_dn(w) - (e_k(k-self.q)+e_k(k))/2
        grn_kmq[0,1] = +(e_k(k-self.q) - e_k(k))/2/1j
        grn_kmq[1,0] = -(e_k(k-self.q) - e_k(k))/2/1j
        grn_kpq = lg.inv(grn_kpq)
        grn_kmq = lg.inv(grn_kmq)
        # GF in global reference frame matrice calculation. Calculation accroding to the (13) formula from the article.
        grn_temp = np.zeros((2,2),dtype = complex)
        grn_temp[0,0] = ( grn_kpq[0,0] + grn_kmq[0,0])
        grn_temp[1,1] = ( grn_kpq[1,1] + grn_kmq[1,1])
        grn_temp[0,1] = (-grn_kpq[0,1] + grn_kmq[0,1])*1j
        grn_temp[1,0] = (+grn_kpq[1,0] - grn_kmq[1,0])*1j
        return -np.imag(grn_temp[0,0]/4+grn_temp[0,1]/4+grn_temp[1,0]/4+grn_temp[1,1]/4)/np.pi

if __name__ == "__main__":
    print("Imports fine.")