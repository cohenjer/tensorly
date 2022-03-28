import tensorly as tl
import numpy as np
from tensorly.random import random_tucker
from tensorly.decomposition._tucker import tucker
from tensorly.tenalg.core_tenalg.n_mode_product import mode_dot
import time
import copy
import plotly.express as px
import pandas as pd

tl.set_backend('numpy')

N_exp = 20

store_pd = pd.DataFrame()
dims_list = [[10,9,11,10,8,12], [100,99,10], [10000,10,5],[5,10000,10],[5,10,10000]]


#dims = dims_list[0]

#a = np.random.randn(3,4,7,8,5)
#b = np.random.randn(2,4)
#d = np.tensordot(a,b,([1],[1]))
#print(d.shape)
#c = mode_dot(a,b,1)
#print(c.shape)

for dims in dims_list:
    for i in range(N_exp):

        gt = random_tucker(dims,len(dims)*[3])
        tensor = gt.to_tensor() #+ 0.001*np.random.randn(*dims)

        ranke = len(dims)*[3]
        init = random_tucker(dims,ranke)

        # Tensordot
        tic = time.time()
        tucker(tensor,ranke,init=copy.deepcopy(init), n_iter_max=50, tol=0, fast=False)
        toc = time.time()-tic
        toc = round(toc,3)
        
        # current
        tic2 = time.time()
        tucker(tensor,ranke, init=copy.deepcopy(init), n_iter_max=50, tol=0, fast="legacy")
        toc2 = time.time()-tic2
        toc2 = round(toc2,3)


        # Checking if output is correct
        print('Run time for tensordot: {}, for legacy: {}'.format(toc,toc2))
        print('')

        dic = {
            'dims': [str(dims),str(dims)],
            'runtime': [toc,toc2],
            'algorithm':['tensordot', 'legacy']
        }
        data = pd.DataFrame(dic)
        store_pd = store_pd.append(data, ignore_index=True)

print(store_pd.groupby('algorithm').mean())

fig = px.box(store_pd, x='dims',y='runtime', color='algorithm', log_y=True, title='HOOI algorithm runtime in Tensorly (rank=[3,3,3], 50 iterations), Numpy Tensordot vs Tensorly naive', width=1904, height=916)
fig.update_layout(
    font=dict(size=18)
)
fig.write_image('./benchmark_tensordot_numpy_tucker.png')
fig.show()