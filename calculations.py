import numpy as np
import h5py
import scipy.linalg
from postDMFT.twoparticle import Chi, SingularPartRight, SingularPartLeft, iQISTResponse, compute_chi_from_phi
from postDMFT.kspace import v2d

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
        "inv_phi" : False,
        "inv_chi" : False,
        "inv_chi_xyz": False,
        "singular_parts": False,
        "U_matrix_extended" : False,
    }

    def __init__(self, h5fn, new_calc_rules = {}):
        # Check if all keys in new_calc_rules are known (present as keys in default_calc_rules).
        for key in new_calc_rules:
            if key not in self.default_calc_rules:
                raise KeyError
        # Merge new_calc_rules and default_calc_rules
        calc_rules = {**self.default_calc_rules, **new_calc_rules}
        self.calc_rules = calc_rules
        self.__check_input()

        # Loading parameters.
        self.__get_params(h5fn)
        bfsym = self.params["symbf"]
        if bfsym:
            self.bfreqs = (np.arange(
                2*self.params["nbfrq"]-1) - self.params["nbfrq"] + 1)*2*np.pi/self.params["beta"]
            self.zero_bfreq = self.params["nbfrq"]-1
        else:
            self.bfreqs = np.arange(
                self.params["nbfrq"])*2*np.pi/self.params["beta"]
            self.zero_bfreq = 0
        self.kmesh = np.linspace(-np.pi,np.pi,self.params["nkp"]*2+1,endpoint=True)
        self.zero_iq = self.params["nkp"]

        # CHI0
        if calc_rules["chi0"]:
            self.chi0 = Chi()
            with h5py.File(h5fn, 'r') as f:
                chi0_data = np.array(f["chi0_closed_real"]).T + \
                    np.array(f["chi0_closed_imag"]).T*1j
            self.chi0.load_from_array(chi0_data)

        # INV_CHI0
        if calc_rules["inv_chi0"]:
            self.inv_chi0 = self.chi0.inv(regularization=0.0)

        # PHI
        if calc_rules["phi"]:
            self.phi = Chi()
            with h5py.File(h5fn, 'r') as f:
                phi_data = np.array(f["phi_real"]).T + np.array(f["phi_imag"]).T*1j
            self.phi.load_from_array(phi_data)
        else:
            # If phi was not loaded use chi0 set phi to chi0.
            self.phi = self.chi0

        # INV_PHI
        if calc_rules["inv_phi"]:
            self.inv_phi = self.phi.inv(regularization=0.0)

        # CHI
        self.chi = compute_chi_from_phi(U_value = self.params["U"], phi = self.phi, regularization = 0.0)

        # INV_CHI
        if calc_rules["inv_chi"]:
            self.inv_chi = self.chi.inv(regularization=0.0)

        # INV_CHI_XYZ
        if calc_rules["inv_chi_xyz"]:
            self.inv_chi_xyz = Chi()
            inv_chi_xyz_data = np.empty_like(self.chi.im_data)
            for iqx in range(inv_chi_xyz_data.shape[0]):
                for iqy in range(inv_chi_xyz_data.shape[1]):
                    for k in range(inv_chi_xyz_data.shape[2]):
                        inv_chi_xyz_data[iqx, iqy, k, :3, :3] = scipy.linalg.inv(self.chi.im_data[iqx, iqy, k, :3, :3])
                        inv_chi_xyz_data[iqx, iqy, k, 3, :] = np.nan
                        inv_chi_xyz_data[iqx, iqy, k, :, 3] = np.nan
            self.inv_chi_xyz.load_from_array(inv_chi_xyz_data)

        # U_MATRIX
        self.U_matrix = np.zeros((4, 4))
        self.U_matrix[ 0,-1] = -self.params["U"]
        self.U_matrix[-1, 0] = -self.params["U"]
        self.U_matrix[ 1, 1] = +self.params["U"]
        self.U_matrix[ 2, 2] = +self.params["U"]

        # U_MATRIX_EXTENDED
        if calc_rules["U_matrix_extended"]:
            self.U_matrix_extended = Chi()
            U_matrix_extended_data = np.empty_like(self.chi.im_data)
            nkp_x, nkp_y, full_nbfrq = self.chi.im_data.shape[:3]
            for iqx in range(nkp_x):
                for iqy in range(nkp_y):
                    for k in range(full_nbfrq):
                        U_matrix_extended_data[iqx, iqy, k, :, :] = self.U_matrix
            self.U_matrix_extended.load_from_array(U_matrix_extended_data)

    def __check_input(self):
        # Check if updated calc_rules are consistent.
        calc_rules = self.calc_rules
        if calc_rules["inv_chi0"] and (not calc_rules["chi0"]):
            raise KeyError("Can not compute inv_chi0 without chi0.")
        if calc_rules["inv_phi"] and (not calc_rules["phi"]):
            raise KeyError("Can not compute inv_phi without phi.")
        if (not calc_rules["phi"]) and (not calc_rules["chi0"]):
            raise KeyError("At least phi or chi0 should be spectidied to be loaded.")

    def recalculate_chi(self,regularization=0.0):
        self.chi = compute_chi_from_phi(U_value = self.params["U"], phi = self.phi, regularization = regularization)
        
    def compute_singular_parts(self, regularization):
        self.singular_part_right = SingularPartRight(U_value=self.params["U"])
        self.singular_part_right.compute_from_phi(self.phi)
        # self.inversed_singular_part_right = self.singular_part_right.inv(regularization)

        self.singular_part_left = SingularPartLeft(U_value=self.params["U"])
        self.singular_part_left.compute_from_phi(self.phi)
        # self.inversed_singular_part_left = self.singular_part_left.inv(regularization)

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