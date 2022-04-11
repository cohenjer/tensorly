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
#dims_list = [[10,9,11,10,8,12],[10,10,10,10], [20,21,22,23], [100,99,101,10], [50,51,52], [200,199,201], [1000,10,11]]
dims_list = [[10,9,11,10,8,12], [20,21,22,20], [64,64,64,12], [50,51,52], [200,200,10], [200,10,200], [10,200,200], [100,5,98,45]]

for dims in dims_list:
    for i in range(N_exp):

        gt = random_cp(dims, 3)
        gt.normalize()
        tensor = gt.to_tensor() #+ 0.001*np.random.randn(*dims)

        ranke = 3
        init = random_cp(dims,ranke)

        # Tensordot
        tic = time.time()
        out = parafac(tensor,ranke,init=copy.deepcopy(init), n_iter_max=50, tol=0, return_errors=True, fast_ttv='tensordot')
        toc = time.time()-tic
        toc = round(toc,3)
        out[0].normalize()
        
        # TTVs subopt
        tic2 = time.time()
        out2 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv='ttv')
        toc2 = time.time()-tic2
        toc2 = round(toc2,3)
        out2[0].normalize()
        
        # TTVs subopt (forward order)
        tic3 = time.time()
        out3 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv='ttvs-forward')
        toc3 = time.time()-tic3
        toc3 = round(toc3,3)
        out3[0].normalize()

        # TTVs subopt (backward order)
        tic4 = time.time()
        out4 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv='ttvs-backward')
        toc4 = time.time()-tic4
        toc4 = round(toc4,3)
        out4[0].normalize()

        # TTVs subopt (optimal order)
        tic5 = time.time()
        out5 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv='ttvs-optimal')
        toc5 = time.time()-tic5
        toc5 = round(toc5,3)
        out5[0].normalize()

        # Current version
        tic6 = time.time()
        out6 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=50, tol=0, fast_ttv="legacy")
        toc6 = time.time()-tic6
        toc6 = round(toc6,3)
        out6[0].normalize()


        # Checking if output is correct
        print('Run time for tensordot: {}, for ttv: {}, for ttvs-f: {}, for ttvs-b: {}, for ttvs-o: {}, for legacy: {}'.format(toc, toc2, toc3, toc4, toc5, toc6))
        print('Final MSE for tensordot: {}, for ttv: {}, for ttvs-f: {}, for ttvs-b: {}, for ttvs-o: {}, for legacy: {}'.format(round(out[1][-1],3),round(out2[1][-1],3), round(out3[1][-1],3),round(out4[1][-1],3),round(out5[1][-1],3), round(out6[1][-1],3)))
        print('')

        dic = {
            'dims': 6*[str(dims)],
            'runtime': [toc, toc2, toc3, toc4, toc5, toc6],
            'recerror': [round(out[1][-1],3), round(out2[1][-1],3), round(out3[1][-1],3), round(out4[1][-1],3), round(out5[1][-1],3), round(out6[1][-1],3)],
            'algorithm':['tensordot', 'ttv', 'ttvs-forward', 'ttvs-backward', 'ttvs-optimal', 'legacy']
        }
        data = pd.DataFrame(dic)
        store_pd = store_pd.append(data, ignore_index=True)

print(store_pd.groupby('algorithm').mean())

fig = px.box(store_pd, x='dims',y='runtime', color='algorithm', log_y=True, title='ALS algorithm runtime in Tensorly (rank=3, 50 iterations) with ttvpy.ttv(s) vs tensordot vs legacy', width=1904, height=916)
fig.update_layout(
    font=dict(size=18)
)
fig.write_image('./benchmark_ttvs_numpy.png')
fig.show()