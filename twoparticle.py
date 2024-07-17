import numpy as np
from postDMFT.kspace import v2d, periodic, BZPath, vBZ
from scipy import interpolate
from scipy import optimize as opt
from postDMFT.common import FLOATZERO
import copy
import scipy.linalg as lg
from postDMFT.utils import spin2index
import ana_cont.continuation as cont
from IPython.utils import io

# Details of ana_cont API and usage can be find: https://josefkaufmann.github.io/ana_cont/api_doc.html and https://arxiv.org/abs/2105.11211 .

class iQISTResponse():
    """
    Class to represent wave-vector and  imaginary frequency -dependent response-like matrix-valued functions.

    Parameters

    Attributes
    ----------
    im_mesh : ndarray
        Storage for nonnegative bosonic imaginary frequency values (without imaginary identity).
    nkp : int
        Number of positive k-points along each axis (total number of k-points is (2*nkp + 1)^2 ).
    nbfrq: int
        Number of bosonic frequencies.
    wv_mesh : ndarray
        k-point mesh along each axis.
    im_data : ndarray
        Complex values of the response function. It's shape is (2*nkp+1,2*nkp+1,nbfrq,4,4).
    interpolated : bool
        Flag where information about performed interpolation is stored.
    """

    def __init__(self):
        self.nbfrq = 0
        self.nkp = 0
        self.wv_mesh = np.linspace(-np.pi, np.pi, 2*self.nkp+1, endpoint=True)
        self.im_data = np.zeros((2*self.nkp+1, 2*self.nkp+1, self.nbfrq, 4, 4), dtype=complex)
        self.interpolated = False


    def load_from_array(self, data: np.ndarray):
        """
        Loads responce function's values from an ndarray and updates k-mesh-dependent structures respectively.

        Parameters
        ----------
            data : ndarray
                Complex values of the response function. It's shape should be (2*nkp+1,2*nkp+1,nbfrq,4,4).
        """
        # Obtaining nkp .
        if data.shape[0] == data.shape[1]:
            self.nkp = (data.shape[0] - 1) // 2
        else:
            raise TypeError("Number of k-points could not be obtained.")
        # Obtaining nbfrq .
        self.nbfrq = data.shape[2]
        # Filling meshes .
        self.wv_mesh = np.linspace(-np.pi, np.pi, self.nkp*2+1, endpoint=True)
        self.im_data = np.zeros(
            (2*self.nkp+1, 2*self.nkp+1, self.nbfrq, 4, 4), dtype=complex)
        if self.im_data.shape == data.shape:
            self.im_data = data.copy()
        else:
            raise TypeError(
                f"Shape of loaded data structure is incompetible with the response function's shape, data shape is {data.shape}.")
        
    def load_from_file(self, nkp: int, nbfrq: int, fn):
        """ Function to load data written in iQIST format from a fn file."""
        im_data = np.zeros((2*nkp+1, 2*nkp+1, nbfrq, 4, 4), dtype=complex)
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
        
    def _check_shape(self,other_response):
        if self.im_data.shape != other_response.im_data.shape:
            raise TypeError("Shapes of summands are incompetible.")
        else:
            return

    def __add__(self, other_response):
        self._check_shape(other_response)
        new_response = iQISTResponse()
        new_response.load_from_array(self.im_data + other_response.im_data)
        return new_response

    def __sub__(self, other_response):
        self._check_shape(other_response)
        new_response = iQISTResponse()
        new_response.load_from_array(self.im_data - other_response.im_data)
        return new_response

    def __matmul__(self, other_response):
        """
        Multiplication of iQISTResponse instances
        point-vise with respect to k-points and imaginary freuencies
        but "matrix-like" with respect to spin multiindexes.
        """
        self._check_shape(other_response)
        new_response = iQISTResponse()
        new_response.load_from_array(np.einsum("lmnij,lmnjk->lmnik",self.im_data,other_response.im_data, optimize='optimal'))
        return new_response

    def __neg__(self):
        new_response = iQISTResponse()
        new_response.load_from_array(-self.im_data.copy())
        return new_response
    
    def inv(self, regularization = 0.0):
        if regularization < 0.0:
            print("Regularizaiton is negative, setting regularization to 'zero'.")
            regularization = FLOATZERO
        inverse_data = self.im_data + np.identity(4)*regularization
        for k in range(self.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    inverse_data[iqx, iqy, k, :, :] = lg.inv(inverse_data[iqx, iqy, k, :, :])
        reginv = iQISTResponse()
        reginv.load_from_array(inverse_data)
        return reginv
        
    def get_regularized_inverse(self, regularization = 0.0):
        # Old method name.
        return self.inv(regularization=regularization)
        

    def interpolate(self, verbose=False, method="RB", method_settings={"Nx": 5, "Ny": 5}):
        """ Method to create functions which interpolate response function on the Brillouin zone."""
        self.interpolated = True
        # As RectBivariateSpline handles only real-valued functions, here i create functions both for real and imaginary parts.
        self.component_functions_re = np.empty(
            (self.nbfrq, 4, 4), dtype=object)
        self.component_functions_im = np.empty(
            (self.nbfrq, 4, 4), dtype=object)
        for kfreq in range(self.nbfrq):
            if verbose:
                print("k = ", kfreq)
            for n in range(4):
                for m in range(4):  
                    if method == "RB":
                        # Extending array to imply periodic conditions on interpolation.
                        dk = self.wv_mesh[1] - self.wv_mesh[0]
                        wv_mesh_extended = np.array([self.wv_mesh[0] - dk,]+list(self.wv_mesh)+[self.wv_mesh[-1] + dk])
                        im_data = self.im_data[:, :, kfreq, n, m]
                        im_data_extended = np.zeros((np.array(im_data.shape) + 2),dtype=complex)
                        for iqx,_ in enumerate(self.wv_mesh):
                            im_data_extended[-1,iqx+1] = im_data[ 0+1,iqx]
                            im_data_extended[ 0,iqx+1] = im_data[-1-1,iqx]
                        for iqy,_ in enumerate(self.wv_mesh):
                            im_data_extended[iqy+1,-1] = im_data[ 0+1,iqy]
                            im_data_extended[iqy+1, 0] = im_data[-1-1,iqy]
                        im_data_extended[-1, 0] = im_data[ 0+1,-1-1]
                        im_data_extended[ 0,-1] = im_data[-1-1, 0+1]
                        im_data_extended[-1,-1] = im_data[ 0+1, 0+1]
                        im_data_extended[ 0, 0] = im_data[-1-1,-1-1]
                        im_data_extended[1:-1,1:-1] = im_data
                        # End of extention
                        self.component_functions_re[kfreq, n, m] = interpolate.RectBivariateSpline(
                            wv_mesh_extended, wv_mesh_extended, np.real(im_data_extended), kx=method_settings["Nx"], ky=method_settings["Ny"])
                        self.component_functions_im[kfreq, n, m] = interpolate.RectBivariateSpline(
                            wv_mesh_extended, wv_mesh_extended, np.imag(im_data_extended), kx=method_settings["Nx"], ky=method_settings["Ny"])
                    else:
                        raise NotImplemented

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
                    tmp_matrix[n, m] = self.component_functions_re[k, n, m](
                        qx, qy) + 1j*self.component_functions_im[k, n, m](qx, qy)
            return tmp_matrix
        else:
            raise RuntimeError("Interpolation wasn't performed yet.")


    def refine_wv_mesh(self, new_nkp: int):
        self.interpolate()
        full_new_nkp = new_nkp*2 + 1
        new_im_data = np.zeros(
            (full_new_nkp, full_new_nkp, self.nbfrq, 4, 4), dtype=complex)
        Kaxis = np.linspace(-np.pi, np.pi, full_new_nkp, endpoint=True)
        for iqx, qx in enumerate(Kaxis):
            for iqy, qy in enumerate(Kaxis):
                for k in range(self.nbfrq):
                    new_im_data[iqx, iqy, k, :, :] = self(v2d(qx, qy), k)
        new_response = iQISTResponse()
        new_response.load_from_array(new_im_data)
        return new_response

def compute_chi_from_phi(U_value: float, phi: iQISTResponse, regularization=0.0):
        signular_part_left = SingularPartLeft(U_value=U_value)
        signular_part_left.compute_from_phi(phi=phi)
        reginv = signular_part_left.inv(regularization=regularization)
        return reginv @ phi

class SingularPart(iQISTResponse):
    def __init__(self, U_value: float):
        super().__init__()
        self.U_matrix = np.zeros((4, 4))
        self.U_matrix[0, 3] = self.U_matrix[3, 0] = -U_value
        self.U_matrix[1, 1] = self.U_matrix[2, 2] = +U_value

    def compute_from_phi(self, phi: iQISTResponse):
        raise NotImplementedError

    def get_eigenvalues(self):
        eigenvalues = np.zeros((self.im_data.shape[:4]),dtype=complex)
        for iqx in range(self.im_data.shape[0]):
            for iqy in range(self.im_data.shape[1]):
                for k in range(self.im_data.shape[2]):
                    eigenvalues[iqx,iqy,k,:] = lg.eigvals(self.im_data[iqx,iqy,k,:,:])
        return eigenvalues

    def get_min_eigenvalues(self):
        eigenvalues = self.get_eigenvalues()
        min_eigenvalues = np.zeros((self.im_data.shape[:3]),dtype=float)
        for iqx in range(self.im_data.shape[0]):
            for iqy in range(self.im_data.shape[1]):
                for k in range(self.im_data.shape[2]):
                    min_eigenvalues[iqx,iqy,k] = np.min(np.real(eigenvalues[iqx,iqy,k,:]))

class SingularPartLeft(SingularPart):
    def compute_from_phi(self, phi: iQISTResponse):
        self.nkp = phi.nkp
        self.nbfrq = phi.nbfrq
        self.wv_mesh = phi.wv_mesh.copy()
        self.im_data = np.empty_like(phi.im_data)
        for k in range(self.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    self.im_data[iqx, iqy, k, :, :] = np.identity(4) - np.matmul(self.im_data[iqx, iqy, k, :, :], self.U_matrix)


class SingularPartRight(SingularPart):
    def compute_from_phi(self, phi: iQISTResponse):
        self.nkp = phi.nkp
        self.nbfrq = phi.nbfrq
        self.wv_mesh = phi.wv_mesh.copy()
        self.im_data = np.empty_like(phi.im_data)
        for k in range(self.nbfrq):
            for iqx in range(2*self.nkp+1):
                for iqy in range(2*self.nkp+1):
                    self.im_data[iqx, iqy, k, :, :] = np.identity(4) - np.matmul(self.U_matrix, self.im_data[iqx, iqy, k, :, :])


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
        Complex values of the response function in spin representation. It's shape is (2*nkp+1,2*nkp+1,nbfrq,3,3).
    """

    sigma_x = np.array([[0.0, 1.0], [1.0,  0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j,  0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sigma_0 = np.array([[1.0, 0.0], [0.0,  1.0]], dtype=complex)
    Sigma = [sigma_x, sigma_y, sigma_z, sigma_0]

    # Matrices to pass from default basis to spin basis.
    TL = np.zeros((4, 4), dtype=complex)
    TR = np.zeros((4, 4), dtype=complex)
    for m in range(4):
        for sigma1 in range(2):
            for sigma2 in range(2):
                alpha = 2*sigma1+sigma2
                TL[m, alpha] = Sigma[m][sigma1, sigma2]/2.0
                TR[alpha, m] = Sigma[m][sigma2, sigma1]/2.0

    # Matrices to pass from spin basis to rotating basis.
    RL = np.matrix(np.zeros((4,4),dtype=complex))
    RL[0,0] = +1j
    RL[0,2] = 1.0
    RL[1,1] = 1.0
    RL[2,0] = -1j
    RL[2,2] = 1.0
    RL[3,3] = 1.0
    RR = RL.T

    def __init__(self):
        super().__init__()
        self.continued = False

    def get_spin_data(self) -> np.ndarray:
        temp = self.im_data.copy()
        shape = list(temp.shape)
        temp_spin = np.zeros(shape, dtype=complex)
        for iqx in range(self.nkp*2+1):
            for iqy in range(self.nkp*2+1):
                for k in range(self.nbfrq):
                    temp_spin[iqx, iqy, k, :,:] = self.TL @ temp[iqx, iqy, k, :, :] @ self.TR
        self.im_data_spin = temp_spin.copy()

    def get_rot_data(self) -> np.ndarray:
        temp = self.im_data.copy()
        shape = list(temp.shape)
        temp_rot = np.zeros(shape, dtype=complex)
        for iqx in range(self.nkp*2+1):
            for iqy in range(self.nkp*2+1):
                for k in range(self.nbfrq):
                    temp_rot[iqx, iqy, k, :,
                              :] = self.RL @ self.TL @ temp[iqx, iqy, k, :, :] @ self.TR @ self.RR
        self.im_data_rot = temp_rot.copy()

    def load_from_array(self, data: np.ndarray):
        super().load_from_array(data)
        self.get_spin_data()
        self.get_rot_data()

    def __add__(self, other_response):
        temp_data = super().__add__(other_response).im_data
        temp_chi = Chi()
        temp_chi.load_from_array(temp_data)
        return temp_chi

    def __matmul__(self, other_response):
        temp_data = super().__matmul__(other_response).im_data
        temp_chi = Chi()
        temp_chi.load_from_array(temp_data)
        return temp_chi
    
    def inv(self, regularization=0.0):
        inv_chi = Chi()
        inv_chi.load_from_array(self.inv(regularization=regularization).im_data)

    def refine_wv_mesh(self,new_nkp: int):
        iqist_response = iQISTResponse()
        iqist_response.load_from_array(self.im_data)
        new_chi = Chi()
        new_chi.load_from_array(iqist_response.refine_wv_mesh(new_nkp=new_nkp).im_data)
        return new_chi

    # def interpolate(self, verbose=False, method="RB", method_settings={ "Nx": 5,"Ny": 5 }):
        # print(method)
        # if method == "phi":
            # raise NotImplemented
            # if method_settings["phi"].interpolated:
                # self.interpolated = True
                # self.phi_interpolation = True
                # self.phi_for_interpolation = copy.deepcopy(method_settings["phi"])
                # self.U_matrix = method_settings["U_matrix"]
                # self.regularization = method_settings["regularization"]
            # else:
                # raise RuntimeError
        # else: 
            # self.phi_interpolation = False
            # return super().interpolate(verbose, method, method_settings)

    # def __call__(self, k: int, qx_i, qy_i, representation="spin"):
        # if self.interpolated:
            # if self.phi_interpolation:
                # phi_calculated = self.phi_for_interpolation(k, qx_i, qy_i, representation="default")
                # tmp_matrix_default = phi_calculated @ lg.inv((1.0 + self.regularization) * np.identity(4) - self.U_matrix @ phi_calculated)
            # else:
                # qx = periodic(qx_i)
                # qy = periodic(qy_i)
                # tmp_matrix_default = np.zeros((4, 4), dtype=complex)
                # for n in range(4):
                    # for m in range(4):
                        # tmp_matrix_default[n, m] = self.component_functions_re[k, n, m](qx, qy) + 1j*self.component_functions_im[k, n, m](qx, qy)
            # if representation == "default":
                # return tmp_matrix_default
            # else:
                # if representation == "spin":
                    # tmp_matrix_spin = self.TL @ tmp_matrix_default @ self.TR
                    # return tmp_matrix_spin
                # elif representation == "rotational":
                    # tmp_matrix_rot = self.RL @ self.TL @ tmp_matrix_default @ self.TR @ self.RR
                    # return tmp_matrix_rot
                # else:
                    # raise TypeError(
                        # "Unrecognized susceptibility's representation type.")
        # else:
            # raise RuntimeError("Interpolation was not performed correctly.")

    # def imax_data(self,q: np.ndarray, representation="spin"):
        # qx = q[0]
        # qy = q[1]
        # temp = np.zeros((len(self.bfreqs),4,4),dtype=complex)
        # for k in range(len(temp)):
            # temp[k] = self(k, qx, qy, representation=representation)
        # return temp

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
            im_data = np.zeros(self.nbfrq, dtype=float)
            for k in range(self.nbfrq):
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