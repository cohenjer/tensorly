import numpy as np
from .core import Backend
import scipy.special
# special gift from Cem Bassoy
try:
    import ttvpy as tp
except:
    print('No local ttvpy package detected')

class NumpyBackend(Backend, backend_name='numpy'):

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def dot(a, b):
        return a.dot(b)
    
    @staticmethod
    def tensordot(a, b, axes=2, fast='tensordot', **kwargs):
        # name axes is not uniform across backends, thus the redefinition
        # also the opportunity here to use faster contraction by bypassing numpy tensordot if available
        # todo: check toolbox availability, better coding, more checks...
        if b.ndim == 1 and fast[:3]=='ttv':
            # accepting ttv and ttvs
            # we are performing tensor times vector contraction
            # processing axes
            mode = axes[0][0]+1 # this supposes that tensordot for ttv is called as tensordot(a,b,([mode],[0]))
            # ttv needs indexing starting from 1
            return tp.ttv(mode,a,b)
        elif b.ndim ==1:
            return np.tensordot(a,b,axes=axes)
        elif b.ndim ==2 and len(axes[0])==1:
            # TTM
            # check if axes is provided
            mode = axes[0][0]
            out = np.tensordot(a,b,axes=axes)
            # tensordot return the free mode on the last mode; we want it to be in place of the contracted one
            order = [i for i in range(out.ndim)]
            order[mode] = out.ndim - 1
            for i in range(mode+1,out.ndim):
                order[i]-=1
            return np.transpose(out, order)
        else:
            return np.tensordot(a,b,axes)
    
    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return x, residuals

    def kr(self, matrices, weights=None, mask=None):
        n_columns = matrices[0].shape[1]
        n_factors = len(matrices)

        start = ord('a')
        common_dim = 'z'
        target = ''.join(chr(start + i) for i in range(n_factors))
        source = ','.join(i + common_dim for i in target)
        operation = source + '->' + target + common_dim

        if weights is not None:
            matrices = [m if i else m*self.reshape(weights, (1, -1)) for i, m in enumerate(matrices)]

        m = mask.reshape((-1, 1)) if mask is not None else 1
        return np.einsum(operation, *matrices).reshape((-1, n_columns))*m

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)

    @staticmethod
    def argsort(tensor, axis, descending = False):
        if descending:
            return np.argsort(-1 * tensor, axis=axis)
        else:
            return np.argsort(tensor, axis=axis)

for name in ['int64', 'int32', 'float64', 'float32', 'complex128', 'complex64', 
             'reshape', 'moveaxis', 'any', 'trace',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros', 'flip',
             'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'matmul',
             'all', 'mean', 'sum', 'cumsum', 'count_nonzero', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'conj', 'diag', 'einsum', 'log', 'log2', 'sin', 'cos', 'exp']:
    NumpyBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'svd', 'eigh']:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ['digamma']:
    NumpyBackend.register_method(name, getattr(scipy.special, name))
