import numpy as np
import tensorly as tl
from tensorly import plugins
import time
from copy import deepcopy

# We run many Nonnegative Tucker decompositions and see if it is faster with TTM or not
time_default = []
time_ttm = []
N = 10
dims_r = [12,12,12]
dims = [128,128,128]

for i in range(N):
    print("Problem number",i)
    T = np.random.rand(*dims)
    init_tucker = (
        np.random.rand(*dims_r),
        [np.random.randn(dims[0],dims_r[0]),
         np.random.randn(dims[1],dims_r[1]),
         np.random.randn(dims[2],dims_r[2])]
        )
    plugins.use_default_mode_dot()
    init1 = deepcopy(init_tucker)
    tic = time.perf_counter()
    tl.decomposition.tucker(T, dims_r,n_iter_max=100,tol=0,init=init1, verbose=False)
    time_default.append(time.perf_counter()-tic)
    plugins.use_ttm_bassoy()
    init2 = deepcopy(init_tucker)
    tic = time.perf_counter()
    tl.decomposition.tucker(T, dims_r,n_iter_max=100,tol=0,init=init2, verbose=False)
    time_ttm.append(time.perf_counter()-tic)