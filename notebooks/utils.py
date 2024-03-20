
from itertools import combinations
import numpy as np
import hoi
import frites
import pickle

def saving(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)
        
def loading(name):
    with open(name,"rb") as f:
        loaded_obj=pickle.load(f)
    return loaded_obj

def goinfo_create_fit(data_brain, beh, **kwmodel, **kwfit):
    
    model = hoi.metrics.GradientOinfo(data_brain, beh, **kwmodel)
    goinfo = model.fit(**kwfit)
    return goinfo

def redundancyMMI_fit(data_brain, beh, **kwmodel, **kwfit):

    model = hoi.metrics.RedundancyMMI(data_brain, beh, **kwmodel)
    goinfo = model.fit(minsize=minsize, maxsize=maxsize, method=method)
    return goinfo

def synergyMMI_fit(data_brain, beh, **kwmodel, **kwfit):

    model = hoi.metrics.SynergyMMI(data_brain, beh, **kwmodel)
    goinfo = model.fit(**kwfit)
    return goinfo

def goinfo_create_fit_correction(data_brain, beh, minsize=3, maxsize=4, **kwmodel, **kwfit):

    model = hoi.metrics.GradientOinfo(data_brain, beh, **kwmodel)
    goinfo = model.fit(**kwfit)

    combos=model.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    list_indices=[[int(c) for c in comb if c != -1] for comb in combos]
    list_multiplets=[str([int(i) for i in comb]) for comb in list_indices]

    #Here we use the function defined in utils to "clean" the higher-order spreading
    goinfo_proc=oinfo_min(goinfo, list_indices, minsize=minsize)

    return goinfo_proc

def task_oinfo_create_fit_correction(data_brain, beh, minsize=2, maxsize=4, method='gcmi'):

    nfeat=data_brain.shape[1]
    yfeat=beh.shape[1]
    modely = hoi.metrics.Oinfo(data_brain, beh)
    oinfoy = modely.fit(minsize=minsize, maxsize=maxsize, method=method)

    model = hoi.metrics.Oinfo(data_brain)
    oinfo = model.fit(minsize=minsize, maxsize=maxsize, method=method)

    oinfo_tot=np.vstack((oinfoy,oinfo))

    combosy=modely.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    combos=-np.ones(combosy.shape)
    #print(combos.shape, combosy.shape)
    combos_=model.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    #print(combos.shape, combos_.shape)
    combos[:combos_.shape[0],:combos_.shape[1]]=combos_
    #print(combos.shape, combosy.shape)

    list_indices=[[c for c in comb if c != -1] for comb in np.vstack((combosy, combos))]
    list_multiplets=[str([int(i) for i in comb]) for comb in list_indices]
    index_h = np.arange(0,len(combosy))

    #Here we use the function defined in utils to "clean" the higher-order spreading
    oinfo_proc=oinfo_min_task(oinfo_tot, list_indices, minsize, nroi=nfeat, ny=yfeat)

    return oinfo_proc[index_h,:]

def rsi_create_fit(data, beh, minsize=2, maxsize=3, method='gcmi'):
    
    model = hoi.metrics.RSI(data, beh)
    res=model.fit(minsize=minsize, maxsize=maxsize, method=method)

    return res

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

def mi_create_fit(data, beh):
    
    list_data_sub = [data,]
    list_beh_sub = [beh,]

    class_final_sub = frites.dataset.DatasetEphy(list_data_sub, y=list_beh_sub, attrs=None)
    wf = frites.workflow.WfMi()
    mi, pvalues = wf.fit(class_final_sub, mcp=None)

    return mi.T

def centrality_nodes(oinf_result, nodes, list_indices, normalized=True):

    n_nodes=len(nodes)
    n_times=oinf_result.shape[1]

    centrality_results=np.zeros((n_nodes, n_times))
    for i in nodes:
        s=np.zeros(n_times)
        cc=0
        for jj, j in enumerate(list_indices):

            if i in j:
                s+=oinf_result[jj,:]
                cc+=1
        if cc != 0:
            if normalized:
                centrality_results[i, :]=s/cc
            else:
                centrality_results[i, :]=s


    return centrality_results

def centrality_create_fit_correction(data_brain, beh, minsize=3, maxsize=4, method='gcmi', normalized=True):

    n_features=data_brain.shape[1]
    model = hoi.metrics.GradientOinfo(data_brain, beh)
    goinfo = model.fit(minsize=minsize, maxsize=maxsize, method=method)

    combos=model.get_combinations(minsize=minsize, maxsize=maxsize)[0]
    list_indices=[[int(c) for c in comb if c != -1] for comb in combos]
    list_multiplets=[str([int(i) for i in comb]) for comb in list_indices]

    #Here we use the function defined in utils to "clean" the higher-order spreading
    goinfo_proc=oinfo_min(goinfo, list_indices, minsize=minsize)
    
    nodes=np.arange(n_features)
    res_cen= centrality_nodes(goinfo_proc, nodes, list_indices, normalized=normalized)
    return res_cen

def bootstrap(data, nboot=1000):

    """the data shuold be given in the form n_samples, n_features"""

    n_sample, n_features = data.shape

    data_boot=np.zeros((n_sample,n_features,nboot))
    for i in range(nboot):
        index=np.random.choice(np.arange(n_sample), n_sample)
        data_boot[:,:,i]=data[index,:]

    return data_boot

