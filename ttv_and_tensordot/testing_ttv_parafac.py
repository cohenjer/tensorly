import tensorly as tl
import numpy as np
from tensorly.random import random_cp
from tensorly.decomposition._cp import parafac
import time
import copy
import plotly.express as px
import pandas as pd

N_exp = 20

store_pd = pd.DataFrame()
dims_list = [[10,9,11,10,8,12],[10,10,10,10], [20,21,22,23], [100,99,101,10], [50,51,52], [200,199,201], [1000,10,11]]

for dims in dims_list:
    for i in range(N_exp):

        gt = random_cp(dims, 3)
        gt.normalize()
        tensor = gt.to_tensor() #+ 0.001*np.random.randn(*dims)

        ranke = 3
        init = random_cp(dims,ranke)

        # Tensordot
        tic = time.time()
        out = parafac(tensor,ranke,init=copy.deepcopy(init), n_iter_max=50, tol=0, return_errors=True, fast_ttv=False)
        toc = time.time()-tic
        toc = round(toc,3)
        out[0].normalize()
        
        # TTV subopt
        tic2 = time.time()
        out2 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv=True)
        toc2 = time.time()-tic2
        toc2 = round(toc2,3)
        out2[0].normalize()
        
        # Current version
        tic3 = time.time()
        out3 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv="legacy")
        toc3 = time.time()-tic3
        toc3 = round(toc3,3)
        out3[0].normalize()


        # Checking if output is correct
        print('Run time for tensordot: {}, for ttv: {}, for legacy: {}'.format(toc,toc2, toc3))
        print('Final MSE for tensordot: {}, for ttv: {}, for legacy: {}'.format(round(out[1][-1],3),round(out2[1][-1],3), round(out3[1][-1],3)))
        print('')

        dic = {
            'dims': [str(dims),str(dims),str(dims)],
            'runtime': [toc,toc2,toc3],
            'recerror': [round(out[1][-1],3),round(out2[1][-1],3),round(out3[1][-1],3)],
            'algorithm':['tensordot', 'ttv', 'legacy']
        }
        data = pd.DataFrame(dic)
        store_pd = store_pd.append(data, ignore_index=True)

print(store_pd.groupby('algorithm').mean())

fig = px.box(store_pd, x='dims',y='runtime', color='algorithm', log_y=True, title='ALS algorithm runtime in Tensorly (rank=3, 50 iterations) with ttvpy.ttv vs tensordot vs legacy', width=1904, height=916)
fig.update_layout(
    font=dict(size=18)
)
fig.write_image('./benchmark_ttv_numpy.png')
fig.show()