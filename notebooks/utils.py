
from itertools import combinations
import numpy as np
import hoi

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

def oinfo_min_task(oinfo_array, list_indices,minsize,nroi,ny=1):

    n_times=oinfo_array.shape[1]
    oi_array=oinfo_array.copy()

    checky= [nroi+i for i in range(ny)]

    for n,m in enumerate(list_indices):

        c=0
        for kkk in m:
            if kkk in checky:
                c+=1
        nn=1
        
        if c>0:
            nn=ny + 1

        if len(m)>=minsize+nn:
            
            sub_ind = []
            for sub_m in combinations(m, len(m)-1):

                c=0
                for kkk in sub_m:
                    if kkk in checky:
                        c+=1

                if c==ny:
                    sub_ind.append(list_indices.index(list(sub_m)))
                
                sub_ind.append(list_indices.index([aaa for aaa in m if aaa not in checky]))
            
            for t in range(n_times):
                diff_oi =[]
                count=0

                for sub_i in sub_ind:
                    count += np.sign(oinfo_array[n,t]-oinfo_array[sub_i,t])
                    diff_oi.append(abs(oinfo_array[n,t]-oinfo_array[sub_i,t]))
                if int(np.abs(count))==len(sub_ind):

                    oi_array[n,t] = min(diff_oi)*np.sign(count)
                
                else:
                    oi_array[n,t]=0

    return oi_array


def goinfo_create_fit(data_brain, beh, minsize=3, maxsize=4):

    model = hoi.metrics.GradientOinfo(data_brain, beh)
    goinfo = model.fit(minsize=minsize, maxsize=maxsize)
    return goinfo

def goinfo_create_fit_correction(data_brain, beh, minsize=3, maxsize=4):

    model = hoi.metrics.GradientOinfo(data_brain, beh)
    goinfo = model.fit(minsize=minsize, maxsize=maxsize)

    combos=model.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    list_indices=[[int(c) for c in comb if c != -1] for comb in combos]
    list_multiplets=[str([int(i) for i in comb]) for comb in list_indices]

    #Here we use the function defined in utils to "clean" the higher-order spreading
    goinfo_proc=oinfo_min(goinfo, list_indices, minsize=minsize)

    return goinfo_proc

def task_oinfo_create_fit_correction(data_brain, beh, minsize=2, maxsize=4):

    nfeat=data_brain.shape[1]
    yfeat=beh.shape[1]
    modely = hoi.metrics.Oinfo(data_brain, beh)
    oinfoy = modely.fit(minsize=minsize, maxsize=maxsize)

    model = hoi.metrics.Oinfo(data_brain)
    oinfo = model.fit(minsize=minsize, maxsize=maxsize)

    oinfo_tot=np.vstack((oinfoy,oinfo))

    combosy=modely.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    combos=-np.ones(combosy.shape)
    #print(combos.shape, combosy.shape)
    combos_=model.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    #print(combos.shape, combos_.shape)
    combos[:combos_.shape[0],:combos_.shape[1]]=combos_
    #print(combos.shape, combosy.shape)

    list_indices=[[int(c) for c in comb if c != -1] for comb in np.vstack((combosy, combos))]
    list_multiplets=[str([int(i) for i in comb]) for comb in list_indices]
    print(list_indices)
    index_h = [i for i, ind in enumerate(list_indices) if nfeat+1 in ind]

    #Here we use the function defined in utils to "clean" the higher-order spreading
    oinfo_proc=oinfo_min_task(oinfo_tot, list_indices, minsize, nroi=nfeat, ny=yfeat)

    return oinfo_proc[index_h,:]

def bootstrap(data, nboot=1000):

    """the data shuold be given in the form n_samples, n_features"""

    n_sample, n_features = data.shape

    data_boot=np.zeros((n_sample,n_features,nboot))
    for i in range(nboot):
        index=np.random.choice(np.arange(n_sample), n_sample)
        data_boot[:,:,i]=data[index,:]

    return data_boot

