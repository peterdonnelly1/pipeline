"""=============================================================================
Dataset-agnostic data loader
============================================================================="""

import math
import numpy as np
import random

from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data         import DataLoader

from   data import GTExV6Config

DEBUG=1

# ------------------------------------------------------------------------------

def get_config(dataset, lr, batch_size ):
    """Return configuration object based on dataset string.
    """

    SUPPORTED_DATASETS = [ 'gtexv6', 'dlbcl', 'eye', 'dlbcl_image',  'mnist']
    
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset %s is not supported.' % dataset)
    if dataset == 'gtexv6':
        print( "LOADER:         INFO:     dataset = \033[35;1m{:}\033[m".format(dataset))
        return GTExV6Config( lr,  batch_size )
    if dataset == 'dlbcl':                                                                                  # PGD 
        print( "LOADER:         INFO:     dataset = \033[35;1m{:}\033[m".format(dataset))
        return GTExV6Config( lr,  batch_size )
    if dataset == 'eye':                                                                                    # PGD SUPPORT ADDED 200125
        print( "LOADER:         INFO:     dataset = \033[35;1m{:}\033[m".format(dataset))
        return GTExV6Config( lr,  batch_size )
    if dataset == 'dlbcl_image':                                                                            # PGD NEW
        print( "LOADER:         INFO:     dataset = \033[35;1m{:}\033[m".format(dataset))
        return GTExV6Config( lr,  batch_size )
    if dataset == 'mnist':
        print( "LOADER:         INFO:     dataset = \033[35;1m{:}\033[m".format(dataset))
        return GTExV6Config( lr,  batch_size )

# ------------------------------------------------------------------------------

def get_data_loaders( cfg, batch_size, num_workers, pin_memory, cv_pct=None, directory=None) :
    
    """Return dataset and return data loaders for train and CV sets.
    """

    if DEBUG>0:
      print( "LOADER:         INFO:   at \033[33;1mget_data_loaders\033[m          with parameters:\
 cfg=\033[35;1m{:}\033[m,\
 batch_size=\033[35;1m{:}\033[m,\
   num_workers=\033[35;1m{:}\033[m,\
      pin_memory=\033[35;1m{:}\033[m,\
      cv_pct=\033[35;1m{:}\033[m,\
     directory=\033[35;1m{:}\033[m"\
.format( cfg, batch_size, num_workers, pin_memory, cv_pct, directory) )

    if cv_pct is not None and directory is not None:
        msg = 'Both CV % and a directory cannot both be specified.'
        raise ValueError(msg)
    if cv_pct is not None and cv_pct >= 1.0:
        raise ValueError('`CV_PCT` should be strictly less than 1.')

    print( "LOADER:         INFO:   about to call dataset specific loader" )
    dataset = cfg.get_dataset()
    print( "LOADER:         INFO:   done" )
    indices = list(range(len(dataset)))

    if directory:
        test_inds  = list(np.load('%s/testset_indices.npy' % directory))
        train_inds = list(set(indices) - set(test_inds))
    else:
        random.shuffle(indices)                                                                            # Shuffles in-place.
        split      = math.floor(len(dataset) * (1 - cv_pct))
        train_inds = indices[:split]
        test_inds  = indices[split:]

    if DEBUG>0:
      print( "LOADER:         INFO:   number of train/test indices      = \033[35;1m{:>5d}, {:>5d}\033[m respectively".format(  len(train_inds), len(test_inds) ) )

    # If batch_size == -1, then we want full batches.
    train_batch_size = batch_size if batch_size != -1 else len(train_inds)
    test_batch_size  = batch_size if batch_size != -1 else len(test_inds)
    assert train_batch_size == test_batch_size

    if DEBUG>0:
      print( "LOADER:         INFO:   train / test batch sizes          = \033[35;1m{:>5d}, {:>5d}\033[m respectively".format(  train_batch_size, test_batch_size ) )
      print( "LOADER:         INFO:   hence number of batches per epoch = \033[35;1m{:>5d}, {:>5d}\033[m respectively".format(  len(train_inds)//train_batch_size, len(test_inds)//test_batch_size ) )

    # If data set size is indivisible by batch size, drop last incomplete batch.
    # Dropping the last batch is fine because we randomly subsample from the
    # data set, meaning all data should be sampled uniformly in expectation.
    DROP_LAST = True

    # This is Soumith's recommended approach. See:
    #
    #     https://github.com/pytorch/pytorch/issues/1106
    #
    
    print( "LOADER:         INFO:   about to create and return data loader for training" )
    train_loader = DataLoader(
        dataset,                                                    # e.g. 'gtexv6
        sampler=SubsetRandomSampler(train_inds),
        batch_size=train_batch_size,                                # from args
        num_workers=num_workers,                                    # from args
        drop_last=DROP_LAST,

        # Move loaded and processed tensors into CUDA pinned memory. See:
        #
        #     http://pytorch.org/docs/master/notes/cuda.html
        #
        pin_memory=pin_memory
    )
    print( "LOADER:         INFO:   train_loader = \033[35;1m{:}\033[m".format(train_loader) )
    
    
    print( "LOADER:         INFO:   about to create and return data loader for testing" )
    test_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(test_inds),
        batch_size=test_batch_size,
        num_workers=num_workers,
        drop_last=DROP_LAST,
        pin_memory=pin_memory
    )
    print( "LOADER:         INFO:   test_loader  = \033[35;1m{:}\033[m".format(test_loader) )
    
    return train_loader, test_loader
