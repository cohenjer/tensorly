import tensorly as tl
import numpy as np
from tensorly.random import random_cp
from tensorly.decomposition._cp import parafac
import time
import copy
import plotly.express as px
import pandas as pd

N_exp = 20
itermax = 50
opt=True

store_pd = pd.DataFrame()
dims_list = [[50,10,500,3], [50,21,49,10],[50,1000,3]]

for dims in dims_list:
    for i in range(N_exp):

        gt = random_cp(dims, 3)
        gt.normalize()
        tensor = gt.to_tensor() + 0.001*np.random.randn(*dims)

        ranke = 3
        init = random_cp(dims,ranke)

        # orderopt
        tic = time.time()
        out = parafac(tensor,ranke,init=copy.deepcopy(init), n_iter_max=itermax, tol=0, return_errors=True, order_opt=opt, fast_ttv="legacy")
        toc = time.time()-tic
        toc = round(toc,3)
        out[0].normalize()
        
        # tensordot orderopt
        tic2 = time.time()
        out2 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=itermax, tol=0, fast_ttv="tensordot", order_opt=opt)
        toc2 = time.time()-tic2
        toc2 = round(toc2,3)
        out2[0].normalize()
        
        # Old tensordot
        tic4 = time.time()
        out4 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=itermax, tol=0, fast_ttv="tensordot", order_opt=False)
        toc4 = time.time()-tic4
        toc4 = round(toc4,3)
        out4[0].normalize()

        # Current version
        tic3 = time.time()
        out3 = parafac(tensor,ranke, init=copy.deepcopy(init), return_errors=True, n_iter_max=itermax, tol=0, fast_ttv="legacy")
        toc3 = time.time()-tic3
        toc3 = round(toc3,3)
        out3[0].normalize()

 

        # Checking if output is correct
        print('Run time for legacy_opt: {}, for legacy: {}, for tensordot_opt: {}, for tensordot_legacy: {}'.format(toc,toc3, toc2, toc4))
        #print('Final MSE for legacy_opt: {}, for tensordot_opt: {}, for ttv_opt: {}, for legacy: {}'.format(round(out[1][-1],3),round(out2[1][-1],3), round(out4[1][-1],3), round(out3[1][-1],3)))
        print('')

        dic = {
            'opt': [opt,opt,opt,False],
            'dims': [str(dims),str(dims),str(dims),str(dims)],
            'runtime': [toc2,toc4,toc,toc3],
            'recerror': [out2[1][-1],out4[1][-1],out[1][-1],out3[1][-1]],
            'algorithm':['tensordot_opt','tensordot', 'legacy_opt', 'legacy']
        }
        data = pd.DataFrame(dic)
        store_pd = store_pd.append(data, ignore_index=True)

print(store_pd.groupby('algorithm').mean())

fig = px.box(store_pd, x='dims',y='runtime', color='algorithm', log_y=True, title='ALS algorithm runtime in Tensorly (rank={}, {} iterations) with Numpy tensordot vs tensorly reshape dot; and modified order vs naive'.format(ranke,itermax), width=1904, height=916)
fig.update_layout(
    font=dict(size=18)
)
fig.write_image('./benchmark_tensordot_orderopt_numpy.png')
fig.show()