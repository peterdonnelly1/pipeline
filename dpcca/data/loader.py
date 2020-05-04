"""=============================================================================
Dataset-agnostic data loader
============================================================================="""

import os
import sys
import math
import random
import torch
import numpy as np

from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data.sampler import SequentialSampler
from   torch.utils.data         import DataLoader

from   data import GTExV6Config

CYAN='\033[36;1m'
RED='\033[31;1m'
PALE_RED='\033[31m'
ORANGE='\033[38;5;136m'
PALE_ORANGE='\033[38;5;172m'
GREEN='\033[32;1m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1

# ------------------------------------------------------------------------------

def get_config( dataset, lr, batch_size ):
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

def get_data_loaders( args, cfg, batch_size, num_workers, pin_memory, pct_test=None, directory=None) :
    
    os.system("taskset -p 0xffffffff %d" % os.getpid())
      
    """Return dataset and return data loaders for train and test sets
    """

    just_test = args.just_test
    
    if DEBUG>0:
      print( "LOADER:         INFO:   at \033[35;1mget_data_loaders\033[m          with parameters:\
 cfg=\033[36;1m{:}\033[m,\
 batch_size=\033[36;1m{:}\033[m,\
   num_workers=\033[36;1m{:}\033[m,\
      pin_memory=\033[36;1m{:}\033[m,\
      pct_test=\033[36;1m{:}\033[m,\
     directory=\033[36;1m{:}\033[m"\
.format( cfg, batch_size, num_workers, pin_memory, pct_test, directory) )

    if pct_test is not None and directory is not None:
        msg = 'Both CV % and a directory cannot both be specified.'
        raise ValueError(msg)
    if pct_test is not None and pct_test > 1.0:
        raise ValueError('`CV_PCT` should be strictly less than 1.')

    print( "LOADER:         INFO:   about to select NN_MODE specific loader" )
    dataset = cfg.get_dataset()
    print( "LOADER:         INFO:       NN_MODE specific loader selected" )
    indices = list(range(len(dataset)))

    if DEBUG>999:
      print( f"LOADER:         INFO:   \033[3mindices  = {indices}" )
      
    if directory:
        test_inds  = list(np.load('%s/testset_indices.npy' % directory))
        train_inds = list(set(indices) - set(test_inds))
    else:
        if just_test=='False':
          random.shuffle(indices)                                                                          # Shuffles in-place.
        split      = math.floor(len(dataset) * (1 - pct_test))
        train_inds = indices[:split]
        test_inds  = indices[split:]

    if DEBUG>0:
      print( f"LOADER:         INFO:       for {CYAN}{pct_test*100:>.0f}%{RESET} split, train/test samples          = {CYAN}{len(train_inds):>6d}{RESET}, {CYAN}{len(test_inds):>5d}{RESET} respectively" )

    train_batch_size = batch_size
    test_batch_size  = batch_size
    assert train_batch_size == test_batch_size

    number_of_train_batches = len(train_inds)//train_batch_size
    number_of_test_batches  = len(test_inds) //test_batch_size
    
    if DEBUG>0:
      print( "LOADER:         INFO:         train / test batch sizes                 = \033[36;1m{:>5d}, {:>5d}\033[m respectively".format(  train_batch_size,         test_batch_size        ) )
      print( "LOADER:         INFO:         number of train / test batches per epoch = \033[36;1m{:>5d}, {:>5d}\033[m respectively".format(  number_of_train_batches,  number_of_test_batches ) )

    if number_of_test_batches<1:
      print( f"{RED}LOADER:         FATAL:      The combination of the chosen batch size and the number of tiles would result in there being zero test batches (you probably need to re-run the tiler) -- halting now{RESET}")
      sys.exit(0)

    # If data set size is indivisible by batch size, drop last incomplete batch.
    # Dropping the last batch is fine because we randomly subsample from the
    # data set, meaning all data should be sampled uniformly in expectation.
    DROP_LAST = True

    # This is Soumith's recommended approach. See:
    #
    #     https://github.com/pytorch/pytorch/issues/1106
    #

    if DEBUG>0:
      print( "LOADER:         INFO:   about to create and return training data loader" )

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
    if DEBUG>99:    
      print( f"LOADER:         INFO:   \033[3mtest_loader  = {CYAN}{train_loader}{RESET}" )

    if DEBUG>9:
      threads=torch.get_num_threads()
      print ( f"{ORANGE}LOADER:         INFO:   number of threads currently being used by Torch = {threads}{RESET}")
    
    if DEBUG>0:
      print( "LOADER:         INFO:   about to create and return test     data loader" )
    
    if just_test=='True':
      print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test' flag is set. Tiles will be selected sequentially rather than at random.{RESET}" )         
      test_loader = DataLoader(
        dataset,
        sampler=SequentialSampler( data_source=dataset ),
        batch_size=test_batch_size,
        num_workers=1,
        drop_last=DROP_LAST,
        pin_memory=pin_memory
    )
    else:
      test_loader = DataLoader(
          dataset,
          sampler=SubsetRandomSampler(test_inds),
          batch_size=test_batch_size,
          num_workers=num_workers,
          drop_last=DROP_LAST,
          pin_memory=pin_memory
      )
    
    if DEBUG>0:    
      print( f"LOADER:         INFO:   \033[3mtest_loader  = {CYAN}{test_loader}{RESET}" )
    
    return train_loader, test_loader
