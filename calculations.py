import numpy as np
import h5py
import scipy.linalg
from postDMFT.twoparticle import Chi, SingularPartRight, SingularPartLeft, iQISTResponse, compute_chi_from_phi, compute_inv_chi_xyz_from_chi
from postDMFT.kspace import v2d, shift_q
from postDMFT.utils import list1d_to_dict, list2d_to_dict

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
            self.compute_inv_chi_xyz()

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
        self.chi = self.compute_chi(regularizaiton=regularization)

    def compute_chi(self,regularization=0.0):
        return compute_chi_from_phi(U_value = self.params["U"], phi = self.phi, regularization = regularization)
    
    def compute_inv_chi_xyz(self, chi: Chi = None) -> None:
        if chi is None:
            chi = self.chi
        self.inv_chi_xyz = Chi()
        inv_chi_xyz_data = compute_inv_chi_xyz_from_chi(chi=chi).im_data
        self.inv_chi_xyz.load_from_array(inv_chi_xyz_data)
        
    def compute_singular_parts(self, regularization=0.0):
        self.singular_part_right = SingularPartRight(U_value=self.params["U"])
        self.singular_part_right.compute_from_phi(self.phi)
        self.inversed_singular_part_right = self.singular_part_right.inv(regularization)

        self.singular_part_left = SingularPartLeft(U_value=self.params["U"])
        self.singular_part_left.compute_from_phi(self.phi)
        self.inversed_singular_part_left = self.singular_part_left.inv(regularization)

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
    
    def load_chi0L(self, h5fn: str) -> None:
        S_strings     = ['tau','x','y']
        T_strings     = ['0','+',  '-']
        T_strings_aux = ['' ,'_p','_m']
        self.chi0L = load_LorR(calc_fn=h5fn, basename="chi0L", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="_closed")

    def load_phiL(self, h5fn: str) -> None:
        S_strings     = ['tau','x','y']
        T_strings     = ['0','+',  '-']
        T_strings_aux = ['' ,'_p','_m']
        self.phiL = load_LorR(calc_fn=h5fn, basename="phiL", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="")

    def load_phiR(self, h5fn: str) -> None:
        S_strings     = ['tau','x','y']
        T_strings     = ['0','+',  '-']
        T_strings_aux = ['' ,'_p','_m']
        self.phiR = load_LorR(calc_fn=h5fn, basename="phiR", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="")

    def load_chi0R(self, h5fn: str) -> None:
        S_strings     = ['tau','x','y']
        T_strings     = ['0','+',  '-']
        T_strings_aux = ['' ,'_p','_m']
        self.chi0R = load_LorR(calc_fn=h5fn, basename="chi0R", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="_closed")

    def load_chi0LR(self, h5fn: str) -> None:
        S_strings = ['tau','x','y']
        T_strings = [None,'+','-']
        T_strings_aux = [None,'p','m']
        self.chi0LR = load_LR(calc_fn=h5fn,basename="chi0LR_", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="_closed")

    def load_phitLR(self, h5fn: str) -> None:
        S_strings = ['tau','x','y']
        T_strings = [None,'+','-']
        T_strings_aux = [None,'p','m']
        self.phitLR = load_LR(calc_fn=h5fn,basename="phitLR_", S_strings=S_strings, T_strings=T_strings, T_strings_aux=T_strings_aux, postfix="")

    def compute_diam(self, h5fn: str) -> None:
        with h5py.File(h5fn,'r') as h5f:
            diam_K_data = get_complex_data(h5f,"diam_K")
        diam_K = np.zeros((3,3),dtype=complex)
        for alpha1 in range(1,3):
            for alpha2 in range(1,3):
                diam_K[alpha1,alpha2]  =  -( diam_K_data[1,alpha1,alpha2,0,0] + diam_K_data[1,alpha1,alpha2,1,1])/4
                diam_K[alpha1,alpha2] +=  -( diam_K_data[2,alpha1,alpha2,0,1] - diam_K_data[2,alpha1,alpha2,1,0])/4*1j
        diam_S_strings = ['x','y']
        self.diam = {}
        for S1_idx, S1 in enumerate(diam_S_strings):
            for S2_idx, S2 in enumerate(diam_S_strings):
                self.diam[f"{S1};{S2}"] = diam_K[1+S1_idx,1+S2_idx]

    def compute_KYY(self, mf_calc: bool = False, regularization: float = 0.0, verbose: bool = False, S_strings = ["+","-"]) -> None:
        T_strings = ["+", "-"]
        K_0          = {}
        K_phi        = {}
        K_ladder     = {}
        K_correction = {}
        def factor_and_spins(ML: list[str,str],MR: list[str,str]) -> list[complex, list[int,int]]:
            if not (len(ML) == 2 and len(MR) == 2): raise RuntimeError
            SL = ML[0]
            TL = ML[1]
            SR = MR[0]
            TR = MR[1]
            f = complex(1.0)
            if SL != "tau":
                f *= (-1j)
            if SR != "tau":
                f *= (-1j)
            if TL == "+":
                sigmaL = 1 # y
                f *= 1.0
            elif TL == "-":
                sigmaL = -1 # 0
                f *= 1.0
            if TR == "+":
                sigmaR = 1 # y
                f *= 1.0
            elif TR == "-":
                sigmaR = -1 # 0
                f *= 1.0
            return f, [sigmaL,sigmaR]
        print("inv_chi_xyz and singular_parts are overwritten by compute_KYY.")
        chi_reg = self.compute_chi(regularization=regularization)
        self.compute_singular_parts(regularization=regularization)
        self.compute_inv_chi_xyz(chi = chi_reg)
        S_strings = ["x","y"]
        T_strings = ["+", "-"]
        for S1 in S_strings:
            for S2 in S_strings:
                key = f"{S1};{S2}"
                K_0[key]          = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_phi[key]        = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_ladder[key]     = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_correction[key] = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                for T1 in T_strings:
                    for T2 in T_strings:
                        f,spins = factor_and_spins([S1,T1],[S2,T2])
                        multikey = f"{S1},{T1};{S2},{T2}"
                        M1 = f"{S1},{T1}"
                        M2 = f"{S2},{T2}"
                        if verbose: print(f,key,multikey,M1,M2,spins)
                        if not mf_calc:
                            K_0[key] += f*(self.chi0LR[multikey].im_data_spin[:,:,:,*spins])
                            K_phi[key] += f*(self.phitLR[multikey].im_data_spin[:,:,:,*spins])
                            K_ladder[key] += f*((self.phiL[M1] @ self.inversed_singular_part_right @ self.U_matrix_extended @ self.phiR[M2]).im_data_spin[:,:,:,*spins])
                            K_correction[key] += f*((self.phiL[M1] @ self.inversed_singular_part_right @ self.inv_chi_xyz @ self.inversed_singular_part_left @ self.phiR[M2]).im_data_spin[:,:,:,*spins])
                        else:
                            K_0[key] += f*(self.chi0LR[multikey].im_data_spin[:,:,:,*spins])
                            K_ladder[key] += f*((self.chi0L[M1] @ self.inversed_singular_part_right @ self.U_matrix_extended @ self.chi0R[M2]).im_data_spin[:,:,:,*spins])
                            K_correction[key] += f*((self.chi0L[M1] @ self.inversed_singular_part_right @ self.inv_chi_xyz @ self.inversed_singular_part_left @ self.chi0R[M2]).im_data_spin[:,:,:,*spins])
        self.KYY_0          = K_0
        self.KYY_phi        = K_phi
        self.KYY_ladder     = K_ladder
        self.KYY_correction = K_correction

    def save_diam(self, calc_fn: str) -> None:
        with h5py.File(calc_fn,'a') as calc_f:
            diam_S_strings = ['x','y']
            if "diam_S_strings" in calc_f:
                del calc_f["diam_S_strings"]
            calc_f.create_dataset("diam_S_strings",data=np.array(diam_S_strings,dtype='S'))
            for S1 in diam_S_strings:
                for S2 in diam_S_strings:
                    key = f"{S1};{S2}"
                    dsetn = f"diam_{key}"
                    if dsetn in calc_f:
                        del calc_f[dsetn]
                    calc_f.create_dataset(dsetn,data=self.diam[key])


    def load_diam(self, calc_fn: str) -> None:
        with h5py.File(calc_fn,'r') as calc_f:
            bdiam_S_strings = np.array(calc_f["diam_S_strings"])
            diam_S_strings = []
            for bkey in bdiam_S_strings:
                key = bkey.decode('ascii')
                diam_S_strings.append(key)
            self.diam = {}
            for S1 in diam_S_strings:
                for S2 in diam_S_strings:
                    key = f"{S1};{S2}"
                    self.diam[key] = complex(np.array(calc_f[f"diam_{key}"]))

    def save_KYY(self,calc_fn: str) -> None:
        with h5py.File(calc_fn,'a') as calc_f:
            if "S_strings" in calc_f:
                del calc_f["S_strings"]
            left_S_strings = set()
            right_S_strings = set()
            for key in self.KYY_0.keys():
                left_S_strings.add(key[0])
                right_S_strings.add(key[-1])
            if not (left_S_strings == right_S_strings): raise RuntimeError
            S_strings = list(left_S_strings)
            calc_f.create_dataset("S_strings",data=np.array(list(S_strings),dtype='S'))
            for S1 in S_strings:
                for S2 in S_strings:
                    key = f"{S1};{S2}"
                    if f"KYY_0_{key}" in calc_f:
                        del calc_f[f"KYY_0_{key}"]
                    calc_f.create_dataset(f"KYY_0_{key}",data=self.KYY_0[key])
                    if f"KYY_phi_{key}" in calc_f:
                        del calc_f[f"KYY_phi_{key}"]
                    calc_f.create_dataset(f"KYY_phi_{key}",data=self.KYY_phi[key])
                    if f"KYY_ladder_{key}" in calc_f:
                        del calc_f[f"KYY_ladder_{key}"]
                    calc_f.create_dataset(f"KYY_ladder_{key}",data=self.KYY_ladder[key])
                    if f"KYY_correction_{key}" in calc_f:
                        del calc_f[f"KYY_correction_{key}"]
                    calc_f.create_dataset(f"KYY_correction_{key}",data=self.KYY_correction[key])

    def save_KXX(self,calc_fn: str) -> None:
        with h5py.File(calc_fn,'a') as calc_f:
            if "S_strings" in calc_f:
                del calc_f["S_strings"]
            left_S_strings = set()
            right_S_strings = set()
            for key in self.KXX_0.keys():
                left_S_strings.add(key[0])
                right_S_strings.add(key[-1])
            if not (left_S_strings == right_S_strings): raise RuntimeError
            S_strings = list(left_S_strings)
            calc_f.create_dataset("S_strings",data=np.array(list(S_strings),dtype='S'))
            for S1 in S_strings:
                for S2 in S_strings:
                    key = f"{S1};{S2}"
                    if f"KXX_0_{key}" in calc_f:
                        del calc_f[f"KXX_0_{key}"]
                    calc_f.create_dataset(f"KXX_0_{key}",data=self.KXX_0[key])
                    if f"KXX_phi_{key}" in calc_f:
                        del calc_f[f"KXX_phi_{key}"]
                    calc_f.create_dataset(f"KXX_phi_{key}",data=self.KXX_phi[key])
                    if f"KXX_ladder_{key}" in calc_f:
                        del calc_f[f"KXX_ladder_{key}"]
                    calc_f.create_dataset(f"KXX_ladder_{key}",data=self.KXX_ladder[key])
                    if f"KXX_correction_{key}" in calc_f:
                        del calc_f[f"KXX_correction_{key}"]
                    calc_f.create_dataset(f"KXX_correction_{key}",data=self.KXX_correction[key])

    def load_KYY(self,calc_fn: str) -> None:
        with h5py.File(calc_fn,'r') as calc_f:
            bS_strings = np.array(calc_f["S_strings"])
            S_strings = []
            for bkey in bS_strings:
                key = bkey.decode('ascii')
                S_strings.append(key)
            self.KYY_0 = {}
            self.KYY_phi = {}
            self.KYY_ladder = {}
            self.KYY_correction = {}
            for S1 in S_strings:
                for S2 in S_strings:
                    key = f"{S1};{S2}"
                    self.KYY_0[key] = np.array(calc_f[f"KYY_0_{key}"])
                    self.KYY_phi[key] = np.array(calc_f[f"KYY_phi_{key}"])
                    self.KYY_ladder[key] = np.array(calc_f[f"KYY_ladder_{key}"])
                    self.KYY_correction[key] = np.array(calc_f[f"KYY_correction_{key}"])
    
    def load_KXX(self,calc_fn: str) -> None:
        with h5py.File(calc_fn,'r') as calc_f:
            bS_strings = np.array(calc_f["S_strings"])
            S_strings = []
            for bkey in bS_strings:
                key = bkey.decode('ascii')
                S_strings.append(key)
            self.KXX_0 = {}
            self.KXX_phi = {}
            self.KXX_ladder = {}
            self.KXX_correction = {}
            for S1 in S_strings:
                for S2 in S_strings:
                    key = f"{S1};{S2}"
                    self.KXX_0[key] = np.array(calc_f[f"KXX_0_{key}"])
                    self.KXX_phi[key] = np.array(calc_f[f"KXX_phi_{key}"])
                    self.KXX_ladder[key] = np.array(calc_f[f"KXX_ladder_{key}"])
                    self.KXX_correction[key] = np.array(calc_f[f"KXX_correction_{key}"])

    def compute_KXX(self, mf_calc: bool = False, regularization: float = 0.0, verbose: bool = False, S_strings = ["+","-"]) -> None:
        T_strings = ["+", "-"]
        K_0          = {}
        K_phi        = {}
        K_ladder     = {}
        K_correction = {}
        def factors_and_spins(ML: list[str,str],MR: list[str,str]) -> list[list[complex], list[list[int,int]]]:
            SL = ML[0]
            TL = ML[1]
            SR = MR[0]
            TR = MR[1]
            if (TL != TR):
                return [complex(0.0)], [[0,0]]
            else:
                fs = []
                spins = []
                if TL == "+":
                    sign = +1.0
                else:
                    sign = -1.0
                for sigma1 in [0,2]:
                    for sigma2 in [0,2]:
                        spin_pair = [sigma1,sigma2] 
                        f = 1.0 / 4.0
                        if SL != "tau":
                            f *= (-1j)
                        if SR != "tau":
                            f *= (-1j)
                        if sigma1 == 0:
                            f *= 1.0
                        elif sigma1 == 2:
                            f *= -1j*sign
                        if sigma2 == 0:
                            f *= 1.0
                        elif sigma2 == 2:
                            f *= +1j*sign
                        fs.append(f)
                        spins.append(spin_pair)
                return fs,spins
        print("inv_chi_xyz and singular_parts are overwritten by compute_KXX.")
        chi_reg = self.compute_chi(regularization=regularization)
        self.compute_singular_parts(regularization=regularization)
        self.compute_inv_chi_xyz(chi = chi_reg)
        S_strings = ["x","y"]
        T_strings = ["+", "-"]
        for S1 in S_strings:
            for S2 in S_strings:
                key = f"{S1};{S2}"
                K_0[key]          = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_phi[key]        = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_ladder[key]     = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                K_correction[key] = np.zeros(self.chi.im_data.shape[:3],dtype=complex)
                for T in T_strings:
                    if T == "+":
                        sign =  1.0
                    else:
                        sign = -1.0
                    multikey = f"{S1},{T};{S2},{T}"
                    M1 = f"{S1},{T}"
                    M2 = f"{S2},{T}"
                    fs,spins = factors_and_spins([S1,T],[S2,T])
                    if verbose: print(key,multikey,M1,M2,fs,spins)
                    if not mf_calc:
                        for f_idx, f in enumerate(fs):
                            spin_pair = spins[f_idx]
                            # K_0
                            array = self.chi0LR[multikey].im_data_spin[:,:,:,*spin_pair] 
                            for k in range(len(self.bfreqs)):
                                K_0[key][:,:,k]            += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                            # K_phi
                            array = self.phitLR[multikey].im_data_spin[:,:,:,*spin_pair] 
                            for k in range(len(self.bfreqs)):
                                K_phi[key][:,:,k]          += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                            # K_ladder
                            array = (self.phiL[M1] @ self.inversed_singular_part_right @ self.U_matrix_extended @ self.phiR[M2]).im_data_spin[:,:,:,*spin_pair]
                            for k in range(len(self.bfreqs)):
                                K_ladder[key][:,:,k]       += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                            # K_correction
                            array = (self.phiL[M1] @ self.inversed_singular_part_right @ self.inv_chi_xyz @ self.inversed_singular_part_left @ self.phiR[M2]).im_data_spin[:,:,:,*spin_pair]
                            for k in range(len(self.bfreqs)):
                                K_correction[key][:,:,k]   += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                    else:
                        for f_idx, f in enumerate(fs):
                            spin_pair = spins[f_idx]
                            # K_0
                            array = self.chi0LR[multikey].im_data_spin[:,:,:,*spin_pair] 
                            for k in range(len(self.bfreqs)):
                                K_0[key][:,:,k]          += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                            # K_ladder
                            array = (self.chi0L[M1] @ self.inversed_singular_part_right @ self.U_matrix_extended @ self.chi0R[M2]).im_data_spin[:,:,:,*spin_pair]
                            for k in range(len(self.bfreqs)):
                                K_ladder[key][:,:,k]     += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
                            # K_correction
                            array = (self.chi0L[M1] @ self.inversed_singular_part_right @ self.inv_chi_xyz @ self.inversed_singular_part_left @ self.chi0R[M2]).im_data_spin[:,:,:,*spin_pair]
                            for k in range(len(self.bfreqs)):
                                K_correction[key][:,:,k] += shift_q(f*(array[:,:,k]),+sign*self.params["Q"],self.kmesh)
        self.KXX_0          = K_0
        self.KXX_phi        = K_phi
        self.KXX_ladder     = K_ladder
        self.KXX_correction = K_correction
       

def get_complex_data(h5f,data_name):
    return (np.array(h5f[data_name+"_real"])+1j*np.array(h5f[data_name+"_imag"])).transpose()

def LorR_to_list(array: np.ndarray) -> list:
    """Convertes numpy array to a list each entity of which is a Chi object.

    Args:
        array (np.ndarray): Asumed to have a shape of (:,:,:,i,:,:), where i marks the axis along which this array is converted to a list.

    Returns:
        list: Has a shape of [i](:,:,:,:,:), where (:,:,:,:,:) is a Chi object.
    """
    if len(array.shape) != 6:
        raise RuntimeError
    n = array.shape[3]
    listarray = [0]*n
    for i in range(n):
        data = array[:,:,:,i,:,:]
        response = Chi()
        response.load_from_array(data)
        listarray[i] = response
    return listarray
    
def LR_to_list(array: np.ndarray) -> list:
    """Convertes numpy array to a nested 2D list each entity of which is a Chi object.

    Args:
        array (np.ndarray): Asumed to have a shape of (:,:,:,i,j,:,:), where i and j mark the axes along which this array is converted to a nested list.

    Returns:
        list: Has a shape of [i][j](:,:,:,:,:), where (:,:,:,:,:) is a Chi object.
    """
    if (len(array.shape) != 7) or (array.shape[3] != array.shape[4]):
        raise RuntimeError
    listarray = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            response = Chi()
            response.load_from_array(array[:,:,:,i,j,:,:])
            listarray[i][j] = response
    return listarray

def LorR_to_dict(array,strings):
    listarray = LorR_to_list(array=array)
    return list1d_to_dict(list1d=listarray,strings=strings)

def LR_to_dict(array,strings):
    listarray = LR_to_list(array=array)
    return list2d_to_dict(list2d=listarray,strings=strings)

## Мультииндексы токовых корреляторов
# Токовые корреляторы отличаются от обычных двухчастичных корреляционных функций
# наличием дополнительных мультииндексов, каждый из которых состоит из двух частей:
# пространственно-временного индекса (тип S, от слова space) и индекса,
# характеризующего тип используемой вершины (тип T, от слова type).
# Все мультииндексы принадлежат одному и тому же множеству мультииндексов,
# представляющему собой декартовое произведение фиксированных множеств индексов типа T и типа S.
# Каждому индексу типа T соответствует некоторая буквенная комбинация. По-умолчанию T \in {tau,x,y}.
# Каждому индексу типа S соответствует некоторая буквенная комбинация. По-умолчанию S \in {  ,+,-}.

## Типы токовых корреляторов
# Существует два типа токовых корреляторов: LorR и LR.
# Первый тип харктеризуется наличием лишь одного мультииндекса, а второй - наличием двух мультииндексов.

## Структура токовых корреляторов, записанных в hdf5 архивы.
# В hdf5 архивах массивы данных, отвечающих токовым корреляторам хранятся в виде нескольких отдельных подмассивов,
# каждый из которых отвечает конкретному индексу (конкретной паре индексов) типа T для
# LorR (LR) токового коррелятора. Конкретный подмассив сохраняется в виде датасета,
# название которого имеед следующий вид.
# Для токового коррелятора типа LorR название датасета {basename}{a}{prefix},
# где a - символьное обозначения индекса типа T.
# Для токового коррелятора типа LR название датасета {basename}{a}{b}{prefix},
# где a,b - символьные обозначения индексов типа T.
# basename и prefix - некоторые наборы символов, возможно включащие в себя символы нижнего подчёркивания.
# Каждый датасет имеет форму
# (K,K,F,S,sigma,sigma) для токового коррелятора типа LorR и
# (K,K,F,S,S,sigma,sigma) для токового коррелятора типа LR,
# где K - индексы, соответствующие волновым векторам, F - частотам, а
# sigma - спиновым индексам.

## Представление в виде структуры данных в Python.
# Токовые корреляторы представлены в Python в виде словаря, состаящего из пар
# ключ - значение, где в качестве значений используются обыкновенные корреляторы типа Chi,
# а каждый ключ соответствует мультиидексу (паре мультииндексов)
# токового коррелятора типа LorR (типа LR).
# В случае токового коррелятора типа LorR отображение ключ -> значение
# имеет вид: "T,S" : Chi
# В случае токового коррелятора типа LR отображение ключ -> значение
# имеет вид: "TL,SL;TR,SR" : Chi

def load_LorR(calc_fn: str, basename: str, S_strings: list, T_strings: list, T_strings_aux: list = None, postfix = "") -> dict:
    """Creates a dictionary representing current correlation function of type LorR.

    Args:
        calc_fn (str): Name of the hdf5 archive.
        basename (str): {basename} used for current correlation function datasets' names.
        S_strings (list): List of S-type strings.
        T_strings (list): List of T-type strings used on Python side.
        T_strings_aux (list, optional): List of T-type strings used on Fortran side. Defaults to S_strings.
        postfix (str, optional): {postfix} used for current correlation function datasets' names. Defaults to "".

    Returns:
        dict: {"S,T": Chi, ...}, where S from S_strings, T from T_strings
    """
    if T_strings_aux is None:
        T_strings_aux = T_strings
    is_ok = True
    is_ok = is_ok and (len(S_strings) == len(T_strings) == len(T_strings_aux))
    if not is_ok:
        raise RuntimeError
    bubble = {}
    with h5py.File(calc_fn,'r') as calc_f:
        for vertex_idx, vertex_string in enumerate(T_strings):
            if vertex_string:
                bubble_data = get_complex_data(calc_f,f"{basename}{T_strings_aux[vertex_idx]}{postfix}")
                bubble_dict = LorR_to_dict(bubble_data,S_strings)
                for st_key in S_strings:
                    bubble[f"{st_key},{vertex_string}"] = bubble_dict[st_key]
    return bubble

def load_LR(calc_fn: str, basename: str, S_strings: list, T_strings: list, T_strings_aux: list = None, postfix = "") -> dict:
    """Creates a dictionary representing current correlation function of  type LR.

    Args:
        calc_fn (str): Name of the hdf5 archive.
        basename (str): {basename} used for current correlation function datasets' names.
        S_strings (list): List of S-type strings.
        T_strings (list): List of T-type strings used on Python side.
        T_strings_aux (list, optional): List of T-type strings used on Fortran side. Defaults to S_strings.
        postfix (str, optional): {postfix} used for current correlation function datasets' names. Defaults to "".

    Returns:
        dict: {"SL,TL;SR,TR": Chi, ...}, where SL,SR from S_strings, TL,TR from T_strings
    """
    if T_strings_aux is None:
        T_strings_aux = T_strings
    is_ok = True
    is_ok = is_ok and (len(S_strings) == len(T_strings) == len(T_strings_aux))
    if not is_ok:
        raise RuntimeError
    bubble = {}
    with h5py.File(calc_fn,'r') as calc_f:
        for vertex_idx_l, vertex_string_l in enumerate(T_strings):
            for vertex_idx_r, vertex_string_r in enumerate(T_strings):
                if vertex_string_l and vertex_string_r:
                    dataset_name = f"{basename}{T_strings_aux[vertex_idx_l]}{T_strings_aux[vertex_idx_r]}{postfix}"
                    bubble_data = get_complex_data(calc_f,dataset_name)
                    bubble_dict = LR_to_dict(bubble_data,S_strings)
                    for st_key_l in S_strings:
                        for st_key_r in S_strings:
                            key = f"{st_key_l},{vertex_string_l};{st_key_r},{vertex_string_r}"
                            bubble[key] = bubble_dict[f"{st_key_l}{st_key_r}"]
    return bubble