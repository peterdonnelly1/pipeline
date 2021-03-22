import h5py
import pandas
import openTSNE
import anndata
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            print ( sep,'-',key,':',obj[key] )
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print ( sep+'\t','-',key,':',obj.attrs[key] )
            dataset_name = f"dataset_{key}"
            dataset_name=h5py.Dataset(key)

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])



h5dump("odata/macosko_2015.h5ad", group='/')




adata =anndata.read_h5ad("odata/macosko_2015.h5ad")
# ~ adata =anndata.read_h5ad("odata/baron_2016h.h5ad")
# ~ adata =anndata.read_h5ad("odata/hrvatin_2018.h5ad")
# ~ adata =anndata.read_h5ad("odata/xin_2016.h5ad ")
# ~ adata =anndata.read_h5ad("odata/chen_2017.h5ad")


affinities = openTSNE.affinity.Multiscale(adata.obsm["X"], perplexities=[50,500], metric="cosine")

init = openTSNE.initialization.pca(adata.obsm["X"])

embedding = TSNEEmbedding(init, affinities)

embedding.optimize(n_iter=250,exaggeration=12, momentum=0.5,inplace=True)

embedding.optimize(n_iter=750,momentum=0.8, inplace=True) 
