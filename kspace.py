import numpy as np
from scipy.linalg import norm
from scipy.interpolate import RegularGridInterpolator

def shift_q(array2d: np.ndarray, q: np.ndarray, K: np.ndarray, method="fft") -> np.ndarray:
    """Method to shift a 2D numerial data defined on a square mesh in K-space on a fixed vector.

    Args:
        array2d (np.ndarray): 2D data to shift.
        q (np.ndarray): Shifting vector.
        K (np.ndarray): 1D mesh of points along each of the axes.
        method (str, optional): If method is "fft", than FFT is used. Else method is passed to the RegularGridInterpolator. Defaults to "fft".

    Returns:
        np.ndarray: Shifted 2D data.
    """
    is_ok = (len(array2d.shape) == 2) and (len(K.shape) == 1) and (array2d.shape[0] == array2d.shape[1] == K.shape[0]) and (len(q.shape) == 1) and (q.shape[0] == 2)
    if not is_ok:
        raise RuntimeError
    if method == "fft":
        return shift_q_2d(array2d, q, K)
    else:
        interp = RegularGridInterpolator((K,K),array2d,method=method)
        new_array2d = np.empty_like(array2d)
        for ix,kx in enumerate(K):
            for iy,ky in enumerate(K):
                k = np.array([kx,ky])
                new_k = k + q
                new_k = periodic(new_k.copy())
                new_array2d[ix,iy] = interp(new_k)[0]
        return new_array2d

def shift_q_1d(data_1d, shift_1d, mesh_1d):
    data_1d_unique = data_1d[:-1]
    mesh_1d_unique = mesh_1d[:-1]
    dx = mesh_1d_unique[1] - mesh_1d_unique[0]
    fft = np.fft.fft(data_1d_unique)
    freqs = np.fft.fftfreq(len(mesh_1d_unique),dx)
    # Note that -1 is used here! f_s(q) = f(q + s) (Это преобразование соответствует сдигу НАЗАД.)
    shifted_fft_unique = fft * np.exp(+2j*np.pi*freqs*shift_1d)
    shifted_data_unique = np.fft.ifft(shifted_fft_unique)
    shifted_data = np.empty_like(data_1d)
    shifted_data[:-1] = shifted_data_unique
    shifted_data[-1] = shifted_data[0]
    return(shifted_data)

def shift_q_2d(data_2d, shift_2d, mesh_1d):
    shifted_data_x = np.empty_like(data_2d)
    for iqy in range(len(mesh_1d)):
        shifted_data_x[:,iqy] = shift_q_1d(data_2d[:,iqy], shift_2d[0], mesh_1d)
    shifted_data_xy = np.empty_like(data_2d)
    for iqx in range(len(mesh_1d)):
        shifted_data_xy[iqx,:] = shift_q_1d(shifted_data_x[iqx,:], shift_2d[1], mesh_1d)
    return shifted_data_xy

def v2d(vx, vy):
    """ Function to easily create 2d wave vectors."""
    v = np.zeros(2)
    v[0] = vx
    v[1] = vy
    return v

def vBZ(name):
    """ Function to easily create 2d wave vectors of high-symmetry points in the first Brillouin zone of a square lattice."""
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

def periodic(qx):
    """ Function to apply periodic conditions on the wave vector component."""
    return (qx + np.pi) % (2*np.pi) - np.pi

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
                length += norm(self.mesh[idx]-self.mesh[idx-1])
                self.T[idx] = length
            for point in points_list[len(self.points_t):]:
                if np.allclose(self.mesh[idx], point):
                    self.points_t.append(self.T[idx])
                    self.points_t_idx.append(idx)
                    break
        self.points_t = np.array(self.points_t)

    def __len__(self):
        return len(self.T)
    
def extend_by_periodicity(array_1d, array_2d):
     # Extending array to imply periodic conditions on interpolation.
    if (len(array_1d.shape) != 1 or len(array_2d.shape) != 2): raise RuntimeError
    if (array_1d.shape[0] != array_2d.shape[0]): raise RuntimeError
    if (array_2d.shape[0] != array_2d.shape[1]): raise RuntimeError("Only square arrays are assumed.")
    npoints = array_1d.shape[0]
    dk = array_1d[1] - array_1d[0]
    array_1d_extended = np.array([array_1d[0] - dk,]+list(array_1d)+[array_1d[-1] + dk])
    array_2d_extended = np.zeros((npoints+2,npoints+2),dtype=array_2d.dtype)
    for iqx,_ in enumerate(array_1d):
        array_2d_extended[-1,iqx+1] = array_2d[ 0+1,iqx]
        array_2d_extended[ 0,iqx+1] = array_2d[-1-1,iqx]
    for iqy,_ in enumerate(array_1d):
        array_2d_extended[iqy+1,-1] = array_2d[ 0+1,iqy]
        array_2d_extended[iqy+1, 0] = array_2d[-1-1,iqy]
    array_2d_extended[-1, 0] = array_2d[ 0+1,-1-1]
    array_2d_extended[ 0,-1] = array_2d[-1-1, 0+1]
    array_2d_extended[-1,-1] = array_2d[ 0+1, 0+1]
    array_2d_extended[ 0, 0] = array_2d[-1-1,-1-1]
    array_2d_extended[1:-1,1:-1] = array_2d
    return array_1d_extended, array_2d_extended