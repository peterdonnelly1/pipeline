"""=============================================================================
Dataset-agnostic data loader
============================================================================="""

import os
import sys
import math
import random
import torch
import numpy as np
import pickle

from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data.sampler import SequentialSampler
from   torch.utils.data         import DataLoader

from   data import GTExV6Config
from   data import MnistConfig
from   data import pre_compressConfig

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

DEBUG=1

# ------------------------------------------------------------------------------

def get_config( dataset, lr, batch_size ):
  
    """Return configuration object based on dataset string.
    """

    SUPPORTED_MODES = [ 'gtexv6', 'dlbcl', 'eye', 'dlbcl_image', 'pre_compress', 'analyse_data', 'mnist']
    
    if dataset not in SUPPORTED_MODES:
        raise ValueError('Dataset %s is not supported.' % dataset)

    if DEBUG>2:
      print( "LOADER:         INFO:     nn_mode = \033[35;1m{:}\033[m".format(dataset))
    if dataset == 'gtexv6':
        return GTExV6Config( )
    elif dataset == 'dlbcl':
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'eye':                                                                                    # PGD SUPPORT ADDED 200125
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'dlbcl_image':                                                                            # PGD NEW
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'pre_compress':                                                                           # PGD SUPPORT ADDED 200713
        return pre_compressConfig( lr,  batch_size )
    elif dataset == 'analyse_data':                                                                           # PGD SUPPORT ADDED 200721
        return pre_compressConfig( lr,  batch_size )                                                        # uses the pre_compress() config file
    elif dataset == 'mnist':
        return MnistConfig()

# ------------------------------------------------------------------------------

def get_data_loaders( args, gpu, cfg, world_size, rank, batch_size, num_workers, pin_memory, pct_test, directory=None) :
    
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
      
    """Return dataset and return data loaders for train and test sets
    """
    
    just_test = args.just_test

    if just_test=='True':
      pct_test=1.0

    if pct_test is not None and directory is not None:
        msg = 'Both CV % and a directory cannot both be specified.'
        raise ValueError(msg)
    if pct_test is not None and pct_test > 1.0:
        raise ValueError('`pct_test` should be  <= 1.')

    if DEBUG>4:
      print( f"{RESET}LOADER:         INFO:     about to select dataset" )
    dataset = cfg.get_dataset( args, gpu )
    if DEBUG>4:    
      print( f"LOADER:         INFO:     dataset loaded" )
    indices = list(range(len(dataset)))
    
    
    if DEBUG>4:
      print( f"LOADER:         INFO:   indices  = {MIKADO}{indices}{RESET}" )
      
    if directory:
        test_inds  = list(np.load('%s/testset_indices.npy' % directory))
        train_inds = list(set(indices) - set(test_inds))
    else:
        if just_test=='False':
          random.shuffle(indices)                                                                          # Shuffles in-place.
        split      = math.floor(len(dataset) * (1 - pct_test))
        train_inds = indices[:split]
        test_inds  = indices[split:]



    # save indices used during training for later use in test mode (so that the same held-out samples will be used for testing in either case)
    if just_test=='False':                                                                                 # we are in training mode, so save ALL indices for possible later use in test mode
      
      if DEBUG>0:
        print ( f"LOADER:         INFO:     train_inds.type         = {PINK}{type(train_inds)}{RESET}"         )
        print ( f"LOADER:         INFO:     train_inds              = {PINK}{train_inds}{RESET}"               )
      if args.input_mode == 'image':
        fqn = f"{args.data_dir}/train_inds_image"
        if DEBUG>0:
              print ( f"LOADER:         INFO:     about to save train_inds to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump( train_inds, f )
      elif args.input_mode == 'rna':
        fqn = f"{args.data_dir}/train_inds_rna"
        if DEBUG>0:
          print ( f"LOADER:         INFO:     about to save train_inds  to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump(train_inds, f)
      elif args.input_mode == 'image_rna':
        fqn = f"{args.data_dir}/train_inds_image_rna"
        if DEBUG>0:
          print ( f"LOADER:         INFO:     about to save train_inds  to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump(train_inds, f)
          
      if DEBUG>0:
          print ( f"LOADER:         INFO:     test_inds.type         = {BLEU}{type(test_inds)}{RESET}"         )
          print ( f"LOADER:         INFO:     test_inds              = {BLEU}{test_inds}{RESET}"               )
      if args.input_mode == 'image':
        fqn = f"{args.data_dir}/test_inds_image"
        if DEBUG>0:
            print ( f"LOADER:         INFO:     about to save test_inds to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump( test_inds, f )
      elif args.input_mode == 'rna':
        fqn = f"{args.data_dir}/test_inds_rna"
        if DEBUG>0:
            print ( f"LOADER:         INFO:     about to save test_inds  to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump(test_inds, f)
      elif args.input_mode == 'image_rna':
        fqn = f"{args.data_dir}/test_inds_image_rna"
        if DEBUG>0:
            print ( f"LOADER:         INFO:     about to save test_inds  to = {MAGENTA}{fqn}{RESET} for possible later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
        with open(fqn, 'wb') as f:
          pickle.dump(test_inds, f)
          
    else:            
                                                                           # if we are in test mode and args.multimode is image_rna retrieve and use the TRAINING indices that were used during unimodal training
      if args.multimode == 'image_rna':

        if DEBUG>0:
            print ( f"{ORANGE}LOADER:         NOTE:     {MAGENTA}'JUST_TEST'{RESET}{PURPLE} and {MAGENTA}args.multimode == 'image_rna'{RESET}. Will load TRAINING indices (only) used during the last unimodal training run{RESET}"         )
              
        if args.input_mode == 'image':
          fqn = f"{args.data_dir}/train_inds_image"
          if DEBUG>0:
            print ( f"LOADER:         INFO:     about to load train_inds from = {MAGENTA}{fqn}{RESET}"         )
          with open(fqn, 'rb') as f:
            test_inds = pickle.load(f)
            if DEBUG>0:
                print ( f"LOADER:         INFO:     test_inds.type         = {PINK}{type(test_inds)}{RESET}"         )
                print ( f"LOADER:         INFO:     test_inds              = {PINK}{test_inds}{RESET}"               )
        elif args.input_mode == 'rna':
          fqn = f"{args.data_dir}/train_inds_rna"
          if DEBUG>0:
            print ( f"LOADER:         INFO:     about to load train_inds  from = {MAGENTA}{fqn}{RESET}"         )
          with open(fqn, 'rb') as f:
            test_inds = pickle.load(f)
            if DEBUG>0:
                print ( f"LOADER:         INFO:     test_inds.type         = {BLEU}{type(test_inds)}{RESET}"         )
                print ( f"LOADER:         INFO:     test_inds              = {BLEU}{test_inds}{RESET}"               )
        elif args.input_mode == 'image_rna':
          fqn = f"{args.data_dir}/train_inds_image_rna"
          if DEBUG>0:
            print ( f"LOADER:         INFO:     about to load train_inds  from = {MAGENTA}{fqn}{RESET}"         )
          with open(fqn, 'rb') as f:
            test_inds = pickle.load(f)
            if DEBUG>0:
                print ( f"LOADER:         INFO:     test_inds.type         = {ARYLIDE}{type(test_inds)}{RESET}"         )
                print ( f"LOADER:         INFO:     test_inds              = {ARYLIDE}{test_inds}{RESET}"               )
  


    train_batch_size = batch_size
    test_batch_size  = batch_size
    assert train_batch_size == test_batch_size

    if DEBUG>0:
      print( f"LOADER:         INFO:                                         train   test"               )
      print( f"LOADER:         INFO:                       mini-batch size: {MIKADO}{train_batch_size:>6d}, {test_batch_size:>5d}{RESET}"               )
      print( f"LOADER:         INFO:              for {MIKADO}{pct_test*100:>3.0f}%{RESET} split, examples: {MIKADO}{len(train_inds):>6d}, {len(test_inds):>5d}{RESET}" )


    number_of_train_batches = len(train_inds)//train_batch_size
    number_of_test_batches  = len(test_inds) //test_batch_size
    
    if DEBUG>0:
      print( f"LOADER:         INFO:                mini-batches per epoch: {MIKADO}{number_of_train_batches:>6d}, {number_of_test_batches:>5d}{RESET}" )
      print( f"LOADER:         INFO:")

    if not just_test=='True':
      if number_of_train_batches<1:
        print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TRAINING batches (consider re-running the tiler or reducing 'BATCH_SIZE' (currently {CYAN}{test_batch_size}{RESET}{RED}) or REDUCING 'PCT_TEST' (currently {CYAN}{pct_test}{RESET}){RED} ) -- halting now{RESET}")
        sys.exit(0)

    if number_of_test_batches<1:
        print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TEST batches (consider re-running the tiler or reducing 'BATCH_SIZE' (currently {CYAN}{test_batch_size}{RESET}{RED}) or REDUCING 'PCT_TEST' (currently {CYAN}{pct_test}{RESET}){RED} ) -- halting now{RESET}")
        sys.exit(0)

    # If data set size is indivisible by batch size, drop last incomplete batch.
    # Dropping the last batch is fine because we randomly subsample from the
    # data set, meaning all data should be sampled uniformly in expectation.
    DROP_LAST = True

    # This is Soumith's recommended approach. See:
    #
    #     https://github.com/pytorch/pytorch/issues/1106
    #

    if DEBUG>2:
      print( "LOADER:         INFO:   about to create and return train loader" )

    if args.ddp=='False': # Single GPU 
      num_workers    = num_workers
      if DEBUG>2:
        print ( f"LOADER:         INFO:     num_workers         = {MIKADO}{num_workers}{RESET}"                  )
      sampler        = SubsetRandomSampler( train_inds )

      train_loader   = DataLoader(
        dataset,
        batch_size   = train_batch_size,
        num_workers  = num_workers,
        sampler      = sampler,
        drop_last    = DROP_LAST,
        pin_memory   = pin_memory                                                                          # Move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
        )        
    else:                 # Multiple GPUs. DistributedSampler will handle dispensing batches to GPUs
      num_workers    = 0
      sampler        = torch.utils.data.distributed.DistributedSampler(                                    # makes sure that each process gets a different slice of the training data
        dataset,
        num_replicas = world_size,
        rank         = rank
        )
      if DEBUG>0:
        print ( f"{BRIGHT_GREEN}LOADER:         INFO:     DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! about to initialize DistributedSampler:{RESET}" )
        print ( f"LOADER:         INFO:       world_size          = {MIKADO}{world_size}{RESET}"          ) 
        print ( f"LOADER:         INFO:       rank                = {MIKADO}{rank}{RESET}"                )
        print ( f"LOADER:         INFO:       num_workers         = {MIKADO}{num_workers}{RESET}"         )
      train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size   = train_batch_size,
        num_workers  = num_workers,
        shuffle      = False,
        sampler      = sampler
        )

    if DEBUG>99:    
      print( f"LOADER:         INFO:   train_loader  = {PURPLE}{train_loader}{RESET}" )

    if DEBUG>9:
      threads=torch.get_num_threads()
      print ( f"{ORANGE}LOADER:         INFO:   number of threads currently being used by Torch = {threads}{RESET}")
    
    if DEBUG>2:
      print( "LOADER:         INFO:   about to create and return test loader" )
    
    if just_test=='True':             
      test_loader = DataLoader(
        dataset,
        #sampler=SequentialSampler( data_source=dataset ),
        sampler  =  SubsetRandomSampler( test_inds ),       
        batch_size=test_batch_size,
        num_workers=1,
        drop_last=DROP_LAST,
        pin_memory=pin_memory
    )
    else:  # just_test=='False' (i.e. training)
      if args.ddp=='False':  # single GPU
        num_workers   = num_workers
        # ~ if just_test=='False':
        sampler  =  SubsetRandomSampler( test_inds )

        if DEBUG>0:
          print ( f"LOADER:         INFO:     training - random sampling will be used{RESET}"                  )          
        # ~ else:
          # ~ sampler  =  SequentialSampler( data_source=dataset )
          # ~ if DEBUG>2:
            # ~ print ( f"LOADER:         INFO:     testing  - sequential sampling will be used{RESET}"               )  
        if DEBUG>2:
          print ( f"LOADER:         INFO:     num_workers         = {MIKADO}{num_workers}{RESET}"                 )
        test_loader = DataLoader(
          dataset,                                                        # e.g. 'gtexv6
          batch_size  = test_batch_size,                                  # from args
          num_workers = num_workers,                                      # from args
          #sampler     = sampler,
          sampler  =  SubsetRandomSampler( test_inds ),              
          drop_last   = DROP_LAST,
          pin_memory  = pin_memory                                                                           # Move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
          )        
      else:                 # Multiple GPUs. DistributedSampler will handle dispensing batches to GPUs
        num_workers    = 0
        sampler        = torch.utils.data.distributed.DistributedSampler(                                           # makes sure that each process gets a different slice of the testing data
          dataset,
          num_replicas = world_size,
          rank         = rank
          )
        if DEBUG>0:
          print ( f"{BRIGHT_GREEN}LOADER:         INFO:     DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! about to initialize DistributedSampler:{RESET}" )
          print ( f"LOADER:         INFO:       world_size          = {MIKADO}{world_size}{RESET}"          ) 
          print ( f"LOADER:         INFO:       rank                = {MIKADO}{rank}{RESET}"                )
          print ( f"LOADER:         INFO:       num_workers         = {MIKADO}{num_workers}{RESET}"         )
        test_loader = torch.utils.data.DataLoader(
          dataset, 
          batch_size              = test_batch_size,
          num_workers             = num_workers,
          shuffle                 = False,
          sampler                 = sampler
          )

    if args.input_mode=='image':
      final_test_batch_size = args.final_test_batch_size
    elif args.input_mode=='rna':
      final_test_batch_size  =  len(test_inds)
    elif args.input_mode=='image_rna':
      final_test_batch_size  =  len(test_inds)


    num_workers            =  num_workers
    final_test_loader = DataLoader(
      dataset,
      batch_size  = final_test_batch_size,
      num_workers = num_workers,
      sampler     = SubsetRandomSampler( test_inds ),
      drop_last   = DROP_LAST,
      pin_memory  = pin_memory                                                                           # Move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
      )      



    if DEBUG>99:    
      print( f"LOADER:         INFO:   test_loader  = {PURPLE}{test_loader}{RESET}" )
    
    torch.cuda.empty_cache()
      
    return train_loader, test_loader, final_test_batch_size, final_test_loader
