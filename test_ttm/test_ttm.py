import numpy as np
import tensorly as tl
import ttmpy
from tensorly import plugins
import time
import matplotlib.pyplot as plt

time_default_ttm = []
time_bassoy_ttm = []
time_default_ttms = []
time_bassoy_ttms = []
time_default_ttv = []
time_bassoy_ttv = []
time_default_ttvs = []
time_bassoy_ttvs = []
N = 50
dims_m = 64
dims = [64,64,12,64]
mode = 1

# TTV
print("Testing TTV")
for i in range(N):
    plugins.use_default_mode_dot()
    A = np.random.randn(dims_m)
    B = np.random.randn(*dims)

    tic = time.perf_counter()
    out = tl.tenalg.mode_dot(B,A,mode)
    time_default_ttv.append(time.perf_counter() - tic)

    plugins.use_ttm_bassoy()
    tic = time.perf_counter()
    out2 = tl.tenalg.mode_dot(B,A,mode)
    time_bassoy_ttv.append(time.perf_counter() - tic)

# TTM
print("Testing TTM")
for i in range(N):
    plugins.use_default_mode_dot()
    A = np.random.randn(dims_m, dims_m)
    B = np.random.randn(*dims)

    tic = time.perf_counter()
    out = tl.tenalg.mode_dot(B,A,mode)
    time_default_ttm.append(time.perf_counter()-tic)

    plugins.use_ttm_bassoy()
    tic = time.perf_counter()
    out2 = tl.tenalg.mode_dot(B,A,mode)
    time_bassoy_ttm.append(time.perf_counter()-tic)

# TTVs
print("Testing TTVs")
for i in range(N):
    A = np.random.randn(dims[0])
    C = np.random.randn(dims[1])
    D = np.random.randn(dims[3])
    B = np.random.randn(*dims)

    plugins.use_default_mode_dot()
    tic = time.perf_counter()
    out = tl.tenalg.multi_mode_dot(B,[A,C,D], skip=2)
    time_default_ttvs.append(time.perf_counter()-tic)

    plugins.use_ttm_bassoy()
    tic = time.perf_counter()
    out2 = tl.tenalg.multi_mode_dot(B,[A,C,D], skip=2)
    time_bassoy_ttvs.append(time.perf_counter()-tic)
    
# Multimode dot
print("Testing Multi Mode Dot")
for i in range(N):
    A = np.random.randn(dims_m,dims[0])
    C = np.random.randn(dims_m,dims[1])
    D = np.random.randn(dims_m,dims[3])
    B = np.random.randn(*dims)

    plugins.use_default_mode_dot()
    tic = time.perf_counter()
    out = tl.tenalg.multi_mode_dot(B,[A,C,D], skip=2)
    time_default_ttms.append(time.perf_counter()-tic)

    plugins.use_ttm_bassoy()
    tic = time.perf_counter()
    out2 = tl.tenalg.multi_mode_dot(B,[A,C,D], skip=2)
    time_bassoy_ttms.append(time.perf_counter()-tic)
    
    
print("Tot time and std for default TTV, TTM, TTMs", sum(time_default_ttv), sum(time_default_ttm), sum(time_default_ttvs), sum(time_default_ttms))
print("Tot time and std for bassoy TTV, TTM, TTMs", sum(time_bassoy_ttv), sum(time_bassoy_ttm), sum(time_bassoy_ttvs), sum(time_bassoy_ttms))


plt.figure()
plt.subplot(2,4,1)
plt.title('Runtime default TTV')
plt.hist(time_default_ttv)
plt.subplot(2,4,2)
plt.title('Runtime default TTM')
plt.hist(time_default_ttm)
plt.subplot(2,4,3)
plt.hist(time_default_ttvs)
plt.title('Runtime default TTVs')
plt.subplot(2,4,4)
plt.title('Runtime default TTM loop')
plt.hist(time_default_ttms)
plt.subplot(2,4,5)
plt.title('Runtime Bassoy TTV')
plt.hist(time_bassoy_ttv)
plt.subplot(2,4,6)
plt.title('Runtime Bassoy TTM')
plt.hist(time_bassoy_ttm)
plt.subplot(2,4,7)
plt.title('Runtime Bassoy TTVs')
plt.hist(time_bassoy_ttvs)
plt.subplot(2,4,8)
plt.title('Runtime Bassoy TTM loop')
plt.hist(time_bassoy_ttms)

rel_time_ttv = (np.array(time_default_ttv)-np.array(time_bassoy_ttv))/np.array(time_default_ttv)
rel_time_ttvs = (np.array(time_default_ttvs)-np.array(time_bassoy_ttvs))/np.array(time_default_ttvs)
rel_time_ttm = (np.array(time_default_ttm)-np.array(time_bassoy_ttm))/np.array(time_default_ttm)
rel_time_ttms = (np.array(time_default_ttms)-np.array(time_bassoy_ttms))/np.array(time_default_ttms)
plt.figure()
plt.subplot(4,1,1)
plt.title('Relative improvements for TTV')
plt.stem(np.sort(rel_time_ttv))
plt.subplot(4,1,2)
plt.title('Relative improvements for TTM')
plt.stem(np.sort(rel_time_ttm))
plt.subplot(4,1,3)
plt.title('Relative improvements for TTVs')
plt.stem(np.sort(rel_time_ttvs))
plt.subplot(4,1,4)
plt.title('Relative improvements for TTM loop')
plt.stem(np.sort(rel_time_ttms))
plt.show()