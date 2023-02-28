import numpy as np
import ana_cont.continuation as cont
from triqs.gf import GfImFreq
from IPython.utils import io

class SystemParameters():
    def __init__(self,path='./',verbose=False):
        # solver.nmat999.dat
        file = open(path + "solver.nmat999.dat")
        for i in range(1):
            line = file.readline()
        self.n_up = float(file.readline().split()[1])
        self.n_dn = float(file.readline().split()[1])
        for i in range(2):
            line = file.readline()
        self.n = float(file.readline().split()[1])
        self.x = 1 - self.n
        file.close()
        # solver.ctqmc.in
        file = open(path + "solver.ctqmc.in")
        for line in file:
            words = line.split()
            if words != [] and words[0] == "mune":
                self.mu = float(words[-1])
            if words != [] and words[0] == "U":
                self.U = float(words[-1])
            if words != [] and words[0] == "beta":
                self.beta = float(words[-1])
            if words != [] and words[0] == "nffrq1":
                self.nffrq = int(words[-1])
            if words != [] and words[0] == "nbfrq1":
                self.nbfrq = int(words[-1])
            if words != [] and words[0] == "Q_x":
                self.Q_x = float(words[-1])
            if words != [] and words[0] == "Q_y":
                self.Q_y = float(words[-1])
        file.close()
        # Suplementary variables and arrays
        self.Q = np.array((self.Q_x,self.Q_y))
        self.delta = (np.pi-self.Q[0])/np.pi
        # self.ffreqs = (2*(np.arange(self.nffrq)-self.nffrq//2)+1)*np.pi/self.beta
        # self.bfreqs = 2*np.arange(self.nbfrq)*np.pi/self.beta
        # Verbos
        if verbose:
            print(f"Q = {self.Q}, delta_Q = {self.delta}")
            print(f"mu = {self.mu:.3f}, n = {self.n:.5f}, x = {self.x:.5f}, n_up = {self.n_up}, n_dn = {self.n_dn}, U = {self.U}, sz = {0.5*(self.n_up-self.n_dn)}")
        
        # def define_bz_mesh(nkp):
            # self.nkp = nkp
            # self.K_x = np.arange(2*self.nkp+1)/(2*self.nkp)*(2*np.pi)-np.pi
            # self.K_y = np.arange(2*self.nkp+1)/(2*self.nkp)*(2*np.pi)-np.pi

def an_continue_general(kernel_mode, im_data, im_freqs, re_freqs, rerror, verbose=False):
    probl = cont.AnalyticContinuationProblem(im_axis=im_freqs, re_axis=re_freqs, im_data=im_data, kernel_mode=kernel_mode)
    model = np.ones_like(re_freqs)
    model /= np.trapz(model, re_freqs)
    err = np.abs(im_data) * rerror
    if verbose:
        sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=model)
    else:
        with io.capture_output():
            sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=model)
    re_data = sol.A_opt
    return re_data

def an_continue_fermionic(im_data, im_freqs, re_freqs, rerror=0.001, verbose=False):
    return an_continue_general('freq_fermionic',im_data, im_freqs, re_freqs, rerror, verbose=verbose)

def an_continue_bosonic(im_data, im_freqs, re_freqs, rerror=0.0001, verbose=False):
    is_real = True
    for i,v in enumerate(im_data):
        if not (np.imag(im_data[i])/abs(im_data[i]) < rerror or np.imag(im_data[i]) < 0.1*rerror):
            is_real = False
            raise ValueError('Data on imaginary axis is not purely real.')
    if is_real:
        return an_continue_general('freq_bosonic',np.real(im_data), im_freqs, re_freqs, rerror, verbose=verbose)*re_freqs
    else:
        return None


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
                                                      # Не имею ни малейшего понятия, что означает этот аргумент.
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

if __name__ == "__main__":
    print("Imports fine.")
