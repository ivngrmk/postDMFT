import numpy as np
import h5py
import scipy.linalg
from twoparticle import Chi, SingularPartRight, SingularPartLeft
from kspace import v2d

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

class Calculation():
    default_calc_rules = {
        "chi0" : False,
        "inv_chi0" : False,
        "phi" : True,
        "Exclude_static_jump": False,
        "inv_phi" : False,
        "U_matrix" : False,
        "inv_chi" : False,
    }

    def __init__(self, h5fn, new_calc_rules = {}):
        # Update default calc_rules with new_calc_rules.
        for key in new_calc_rules:
            if key not in self.default_calc_rules:
                raise KeyError
        calc_rules = {**self.default_calc_rules, **new_calc_rules}
        # Check if updated calc_rules are consistent.
        if calc_rules["inv_chi0"] and (not calc_rules["chi0"]):
            raise KeyError # One can not compute inv_chi0 without chi0.
        if calc_rules["inv_phi"] and (not calc_rules["phi"]):
            raise KeyError # One can not compute inv_phi without phi
        if (not calc_rules["phi"]) and (not calc_rules["chi0"]):
            raise KeyError # At least one of phi or chi0 should be loaded.

        # Loading parameters.
        self.__get_params(h5fn)
        bfsym = self.params["symbf"]
        system = HubbardSystem()
        system.nkp = self.params["nkp"]
        print(system.nkp)
        system.beta = self.params["beta"]
        system.U = self.params["U"]
        self.hsystem = system
        if bfsym:
            system.nbfrq = 2*self.params["nbfrq"]-1
            self.bfreqs = (np.arange(
                2*self.params["nbfrq"]-1) - self.params["nbfrq"] + 1)*2*np.pi/self.params["beta"]
            self.zero_bfreq = self.params["nbfrq"]-1
        else:
            system.nbfrq = self.params["nbfrq"]
            self.bfreqs = np.arange(
                self.params["nbfrq"])*2*np.pi/self.params["beta"]
            self.zero_bfreq = 0
        self.kmesh = np.linspace(-np.pi,np.pi,self.params["nkp"]*2+1,endpoint=True)
        self.zero_iq = self.params["nkp"]

        if calc_rules["chi0"]:
            self.chi0 = Chi(system)
            with h5py.File(h5fn, 'r') as f:
                chi0_data = np.array(f["chi0_closed_real"]).T + \
                    np.array(f["chi0_closed_imag"]).T*1j
            self.chi0.load_from_array(chi0_data)
        if calc_rules["inv_chi0"]:
            self.inv_chi0 = Chi(system)
            inv_chi0_data = np.empty_like(self.chi0.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi0.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        try:
                            inv_chi0_data[iqx, iqy, k, :, :] = scipy.linalg.inv(self.chi0.im_data[iqx, iqy, k, :, :])
                        except scipy.linalg.LinAlgError as err:
                            print(err, iqx,iqy,k)
                            inv_chi0_data[iqx, iqy, k, :, :] = np.zeros((4,4))
            self.inv_chi0.load_from_array(inv_chi0_data)

        self.phi = Chi(system)
        if calc_rules["phi"]:
            with h5py.File(h5fn, 'r') as f:
                phi_data = np.array(f["phi_real"]).T + np.array(f["phi_imag"]).T*1j
        else:
            phi_data = self.chi0.im_data.copy()

        if calc_rules["Exclude_static_jump"]:
            for iqx in range(2*self.params["nkp"]+1):
                for iqy in range(2*self.params["nkp"]+1):
                    for m in range(4):
                        for n in range(4):
                            if np.max(abs(np.real(phi_data[iqx,iqy,:,m,n]))) > 10**(-1):
                                datax = self.bfreqs[self.zero_bfreq+1:]
                                datay = np.real(phi_data[iqx,iqy,self.zero_bfreq+1:,m,n])
                                p = np.polyfit(datax,1 / datay, 3)
                                fit_function = np.vectorize(lambda wn: 1 / np.polyval(p,wn))
                                # def fit_f(wn,A,B,C,D,F):
                                    # return (B + A*wn**2 + D*wn**4)/(1.0 + C*wn**2 + F*wn**4)
                                # popt,_ = scipy.optimize.curve_fit(fit_f,datax,datay,[0.0, datay[0]*10.0, 0.0, datay[-1]*0.5, 1.0])
                                # fit_function = np.vectorize(lambda wn: fit_f(wn,*popt))
                                phi_data[iqx,iqy,self.zero_bfreq,m,n] = fit_function(0.0) + 1j*np.imag(phi_data[iqx,iqy,self.zero_bfreq,m,n])
        self.phi.load_from_array(phi_data)

        if calc_rules["inv_phi"]:
            self.inv_phi = Chi(system)
            inv_phi_data = np.empty_like(self.phi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.phi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        try:
                            inv_phi_data[iqx, iqy, k, :, :] = scipy.linalg.inv(self.phi.im_data[iqx, iqy, k, :, :])
                        except scipy.linalg.LinAlgError as err:
                            print(err, iqx,iqy,k)
                            inv_phi_data[iqx, iqy, k, :, :] = np.zeros((4,4))
            self.inv_phi.load_from_array(inv_phi_data)

        singular_part_right = SingularPartRight(system)
        singular_part_right.compute_from_phi(self.phi)
        inversed_singular_part_right = singular_part_right.get_regularized_inverse(0.0)

        self.inversed_singular_part = inversed_singular_part_right
        self.inversed_singular_part_right = inversed_singular_part_right

        singular_part_left = SingularPartLeft(system)
        singular_part_left.compute_from_phi(self.phi)
        inversed_singular_part_left = singular_part_left.get_regularized_inverse(0.0)

        self.inversed_singular_part_left = inversed_singular_part_left

        self.chi = Chi(system)
        self.chi.load_from_array((self.phi @ inversed_singular_part_right).im_data)

        if calc_rules["inv_chi"]:
            self.inv_chi = Chi(system)
            inv_chi_data = np.empty_like(self.chi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        try:
                            inv_chi_data[iqx, iqy, k, :3, :3] = scipy.linalg.inv(
                                self.chi.im_data[iqx, iqy, k, :3, :3])
                        except scipy.linalg.LinAlgError as err:
                            print(err, iqx,iqy,k)
                            inv_chi_data[iqx, iqy, k, :, :] = np.zeros((4,4))
            self.inv_chi.load_from_array(inv_chi_data)

        self.U_matrix = np.zeros((4, 4))
        self.U_matrix[0, -1] = -self.params["U"]
        self.U_matrix[-1, 0] = -self.params["U"]
        self.U_matrix[1, 1] = self.params["U"]
        self.U_matrix[2, 2] = self.params["U"]

        if calc_rules["U_matrix"]:
            self.extended_U_matrix = Chi(system)
            extended_U_matrix_data = np.empty_like(self.chi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        extended_U_matrix_data[iqx, iqy, k, :, :] = self.U_matrix
            self.extended_U_matrix.load_from_array(extended_U_matrix_data)

    def recalculate_chi(self,source="chi_call"):
        if source == "chi_call":
            if not self.chi.interpolated:
                raise RuntimeError
            chi_data = np.empty_like(self.chi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        chi_data[iqx,iqy,k,:,:] = self.chi(k, self.kmesh[iqx], self.kmesh[iqy], representation="default")
            self.chi.load_from_array(chi_data)
            inv_chi_data = np.empty_like(self.chi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        try:
                            inv_chi_data[iqx, iqy, k, :, :] = scipy.linalg.inv(self.chi.im_data[iqx, iqy, k, :, :])
                        except scipy.linalg.LinAlgError as err:
                            print(err, iqx,iqy,k)
                            inv_chi_data[iqx, iqy, k, :, :] = np.zeros((4,4))
            self.inv_chi.load_from_array(inv_chi_data)
        else:
            raise ValueError("Wrong source parameter.")
        
    def regularize_singular_parts(self, regularization):
        singular_part_right = SingularPartRight(self.hsystem)
        singular_part_right.compute_from_phi(self.phi)
        self.inversed_singular_part = singular_part_right.get_regularized_inverse(regularization)
        self.inversed_singular_part_right = singular_part_right.get_regularized_inverse(regularization)

        singular_part_left = SingularPartLeft(self.hsystem)
        singular_part_left.compute_from_phi(self.phi)
        self.inversed_singular_part_left = singular_part_left.get_regularized_inverse(regularization)

    def __get_params(self, h5fn):
        params = {}
        with h5py.File(h5fn, 'r') as f:
            attrs = f.attrs
            mu = attrs["mu"][0]
            nffrq = attrs["nffrq"][0]
            nbfrq = attrs["nbfrq"][0]
            beta = attrs["beta"][0]
            U = attrs["U"][0]
            try:
                symbf = attrs["symbf"][0] != 0
            except:
                pass
            nup = attrs["n_up"][0]
            ndn = attrs["n_dn"][0]
            sz = (nup - ndn)/2.0
            x = 1 - nup - ndn
            n = nup + ndn
            chi0_data = np.array(f["chi0_closed_real"]).T + \
                np.array(f["chi0_closed_imag"]).T*1j
            nkp = (chi0_data.shape[0]-1) // 2
            Q = v2d(attrs["orderQx"][0],attrs["orderQy"][0])
        params["mu"] = mu
        params["nffrq"] = nffrq
        params["nbfrq"] = nbfrq
        params["beta"] = beta
        params["U"] = U
        params["nup"] = nup
        params["ndn"] = ndn
        params["sz"] = sz
        params["n"] = n
        params["x"] = x
        params["nkp"] = nkp
        params["Q"] = Q
        try:
            params["symbf"] = symbf
        except:
            params["symbf"] = None
        self.params = params