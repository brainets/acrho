
from itertools import combinations
import numpy as np

def oinfo_min(oinfo_array, list_indices,minsize):

    n_times=oinfo_array.shape[1]
    oi_array=oinfo_array.copy()

    for n,m in enumerate(list_indices):

        if len(m)>=minsize+1:

            sub_ind = []
            for sub_m in combinations(m, len(m)-1):
                sub_ind.append(list_indices.index(list(sub_m)))
            
            for t in range(n_times):
                diff_oi =[]
                count=0

                for sub_i in sub_ind:
                    count += np.sign(oinfo_array[n,t]-oinfo_array[sub_i,t])
                    diff_oi.append(abs(oinfo_array[n,t]-oinfo_array[sub_i,t]))
                if int(np.abs(count))==len(m):

                    oi_array[n,t] = min(diff_oi)*np.sign(count)
                
                else:
                    oi_array[n,t]=0

    return oi_array