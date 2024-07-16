import numpy as np
from scipy.linalg import norm

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