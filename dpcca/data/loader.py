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
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
CAMEL='\033[38;2;193;154;107m'
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_config( dataset, lr, batch_size ):
  
    """Return configuration object based on dataset string.
    """

    SUPPORTED_MODES = [ 'gtexv6', 'dlbcl', 'eye', 'dlbcl_image', 'pre_compress', 'analyse_data', 'mnist']
    
    if dataset not in SUPPORTED_MODES:
        raise ValueError('Dataset %s is not supported.' % dataset)

    if dataset == 'gtexv6':
        return GTExV6Config( )
    elif dataset == 'dlbcl':
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'eye':                                                                                 # PGD SUPPORT ADDED 200125
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'dlbcl_image':                                                                         # PGD NEW
        return GTExV6Config( lr,  batch_size )
    elif dataset == 'pre_compress':                                                                        # PGD SUPPORT ADDED 200713
        return pre_compressConfig( lr,  batch_size )
    elif dataset == 'analyse_data':                                                                        # PGD SUPPORT ADDED 200721
        return pre_compressConfig( lr,  batch_size )                                                       # uses the pre_compress() config file
    elif dataset == 'mnist':
        return MnistConfig()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_data_loaders( args, gpu, cfg, world_size, rank, batch_size, num_workers, pin_memory, pct_test, writer, directory=None) :
    
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
      
    """Create and return dataset(s) and data loaders for train and test datasets as appropriate
    """
    
    input_mode             = args.input_mode
    n_tiles                = args.n_tiles
    final_test_batch_size  = args.final_test_batch_size
    just_test              = args.just_test
    use_autoencoder_output = args.use_autoencoder_output


    
    # 1 Preparation

    if DEBUG>2:
      print( f"LOADER:         INFO:    pct_test  = {MIKADO}{pct_test}{RESET}" )
           
    if pct_test is not None and pct_test > 1.0:
        raise ValueError('`pct_test` should be  <= 1.')

    if just_test=='True':
      pct_test=1.0                                                                                         # in test mode, all tiles are test tiles
      


    # 2 Fetch dataset(s)
    
    if DEBUG>2:
      print( f"{RESET}LOADER:         INFO:    about to load dataset(s)" )

    if input_mode=='image':

      if args.cases!='ALL_ELIGIBLE_CASES':                                                                 # i.e. other than 'ALL_ELIGIBLE_CASES' 
        
        # always load the test dataset ... (and if we are in just_test mode, that's all we need)
        which_dataset = 'dataset_image_test'      
        dataset_image_test = cfg.get_dataset( args, which_dataset, gpu )
        # equates via cfg.get_dataset to: dataset = GTExV6Dataset( cfg, which_dataset, args ), i.e. make an object of class GTExV6Dataset using it's __init__() constructor
        # and dataset_image_test.images = data_image_test['images'] etc.; noting that 'data_image_test' is a tensor: see dataset() where data = torch.load(f"data/dlbcl_image/{which_dataset}.pth"
        
        if DEBUG>0  :    
          print( f"LOADER:         INFO:        dataset {CYAN}{which_dataset}{RESET} now loaded" )      
  
        test_inds = list(range(len( dataset_image_test )  )   )

        if DEBUG>80:
          print( f"LOADER:         INFO:    test_inds  = \n{MIKADO}{test_inds}{RESET}" ) 
                  
        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled
          random.shuffle( test_inds )
        
        
        # ... but load the training dataset only if we're in training mode, and use the name 'dataset' (rather than the more obvious 'dataset_image_train') so that it will be compatible with rna mode in subsequent code
        if just_test!='True':
            
          which_dataset = 'dataset_image_train'
          dataset       = cfg.get_dataset( args, which_dataset, gpu )
          # equates via cfg.get_dataset to: dataset = GTExV6Dataset( cfg, which_dataset, args ), i.e. make an object of class GTExV6Dataset using it's __init__() constructor
          # so  dataset.images = data['images'] etc.; noting that 'dataset' is a tensor:  see dataset() where data = torch.load(f"data/dlbcl_image/{which_dataset}.pth"
    
          if DEBUG>0:    
            print( f"LOADER:         INFO:    dataset {CYAN}{which_dataset}{RESET} now loaded"             )      
                
          train_inds = list(range(len( dataset )  )   )  

          if DEBUG>2:
            print( f"LOADER:         INFO:    train_inds  = \n{MIKADO}{train_inds}{RESET}"                 )
            
          random.shuffle(train_inds)                                                                       # in training mode, it's critical that both the training and test sets are shuffled
            
      

      else:     # ALL_ELIGIBLE_CASES


        which_dataset = 'dataset_image_train'      
        # ~ dataset       = cfg.get_dataset( args, which_dataset, writer, gpu )                            # 21-09-23  removed introduced error caused by the presence of 'writer' parameter
        dataset       = cfg.get_dataset( args, which_dataset, gpu )
        # equates via cfg.get_dataset to: dataset = GTExV6Dataset( cfg, which_dataset, args ), i.e. make an object of class GTExV6Dataset using it's __init__() constructor
        # and dataset_image_train.images = dataset_image_train['images'] etc.; noting that 'dataset_image_train' is a tensor:  see dataset() where data = torch.load(f"data/dlbcl_image/{which_dataset}.pth"
        
        if DEBUG>0:    
          print( f"LOADER:         INFO:    dataset {CYAN}{which_dataset}{RESET} now loaded" )      

        indices = list(range( len( dataset )  )   )

        if DEBUG>44:
          print( f"LOADER:         INFO:    indices                         = \n{MIKADO}{indices}{RESET}"      )

        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled ...

          random.shuffle( indices )                                                                        # ... (in test mode, we only use the test indices, and they must not be shuffled as we have to recreate the patches for visualization on Tensorboard)
           
          split      = math.floor(len(indices) * (1 - pct_test))                                   
          train_inds = indices[:split]
          test_inds  = indices[split:]
          
          if DEBUG>44:
            print( f"LOADER:         INFO:    train_inds  ( after shuffle ) = \n{MIKADO}{train_inds}{RESET}" )
            print( f"LOADER:         INFO:    test_inds   ( after shuffle ) = \n{MIKADO}{test_inds}{RESET}"  )

        else:

          if use_autoencoder_output!='True':                                                               # default case (unimode 'just_test' to create patches) (i.e. we are NOT autoencoding as a prelude to clustering

            split      = math.floor(len(indices) * (1 - pct_test))                                   
            train_inds = indices[:split]
            test_inds  = indices[split:]
  
            if DEBUG>44:
              print( f"LOADER:         INFO:    train_inds                  = \n{CARRIBEAN_GREEN}{train_inds}{RESET}" )
              print( f"LOADER:         INFO:    test_inds                   = \n{CARRIBEAN_GREEN}{test_inds}{RESET}"  )
       
          else:                                                                                            # autoencoding as a prelude to clustering 
                                                                                                           # when using an autoencoder, we want to be able to process every tile in test mode, in particular so that we have as many tiles as possible to use when clustering
            test_inds  = indices                                                                           # (we never use ALL_ELIGIBLE_CASES for the multimode scenario; only for unimode and clustering, so this is safe)

            if DEBUG>44:
              print( f"LOADER:         INFO:    test_inds                   = \n{BITTER_SWEET}{test_inds}{RESET}"  )                                                                                      
            


    else:   # rna, image_rna
      
      which_dataset = 'train'
      # ~ dataset = cfg.get_dataset( args, which_dataset, writer, gpu )                                    # 21-09-23  removed introduced error caused by the presence of 'writer' parameter    
      dataset = cfg.get_dataset( args, which_dataset, gpu )
      # equates to dataset = GTExV6Dataset( cfg, args ); i.e. make an object of class GTExV6Dataset using it's __init__()
      # so  dataset.images            = data           ['images'] etc., noting that 'data'            is a tensor:  see dataset() where data = torch.load(f"data/dlbcl_image/{which_dataset}.pth"
      # and dataset_image_test.images = data_image_test['images'] etc., noting that 'data_image_test' is a tensor:  see dataset() where data = torch.load(f"data/dlbcl_image/{which_dataset}.pth"
      
      if DEBUG>2:    
        print( f"LOADER:         INFO:    dataset loaded" )
        
      indices = list(range(len(dataset)))

      if DEBUG>2:
        print( f"LOADER:         INFO:   rna (or image_rna) indices  = {MIKADO}{indices}{RESET}" )
  
      if just_test!='True':                                                                                # in training mode, it's critical that both the training and test sets are shuffled
        random.shuffle(indices)                                                                            # (in test mode, we only use the test indices, and they must not be shuffled as we have to recreate the patches for visualization on Tensorboard)

      split      = math.floor(len(dataset) * (1 - pct_test))
      train_inds = indices[:split]
      test_inds  = indices[split:]
     
    
  

    # 3 maybe save indices used during training for later use in test mode (so that the same held-out samples will be used for testing in either case)
    
    if args.cases=='UNIMODE_CASE____MATCHED': ######################################################### TODO MAKE NICER
    
      if just_test!='True':                                                                                # training mode
  
        #  3A save training indices for possible later use in test
        
        if args.input_mode == 'image':
          if DEBUG>99:
            print ( f"LOADER:         INFO:     (unmodified) train_inds              = {PINK}{train_inds}{RESET}"               )
          fqn = f"{args.data_dir}/train_inds_image"
          if DEBUG>99:
                print ( f"LOADER:         INFO:     about to save train_inds to = {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
          with open(fqn, 'wb') as f:
            pickle.dump( train_inds, f )
  
          if DEBUG>99:
              print ( f"LOADER:         INFO:     (unmodified) test_inds              = {BLEU}{test_inds}{RESET}"               )
          fqn = f"{args.data_dir}/test_inds_image"
          if DEBUG>99:
              print ( f"LOADER:         INFO:     about to save test_inds to = {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
          with open(fqn, 'wb') as f:
            pickle.dump( test_inds, f )
                      
        elif args.input_mode == 'rna':
          if DEBUG>99:
            print ( f"LOADER:         INFO:     (unmodified) train_inds              = {PINK}{train_inds}{RESET}"               )
          fqn = f"{args.data_dir}/train_inds_rna"
          if DEBUG>99:
            print ( f"LOADER:         INFO:     about to save train_inds to: {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
          with open(fqn, 'wb') as f:
            pickle.dump(train_inds, f)
  
          if DEBUG>99:
            print ( f"LOADER:         INFO:     (unmodified) test_inds              = {BLEU}{test_inds}{RESET}"               )
          fqn = f"{args.data_dir}/test_inds_rna"
          if DEBUG>99:
            print ( f"LOADER:         INFO:     about to save test_inds  to: {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"         )
          with open(fqn, 'wb') as f:
            pickle.dump(test_inds, f)

  
      # 3B For 'image' TEST mode and 'rna' TEST mode retrieve and use the TRAINING indices that were used during unimodal training
              
      elif just_test=='True':                                                                              # test mode     
        
        if args.multimode == 'image_rna':
        
          if DEBUG>33:
              print ( f"{CARRIBEAN_GREEN}LOADER:         NOTE:     {MAGENTA}args.just_test == 'True'{RESET}{CARRIBEAN_GREEN} and {MAGENTA}args.multimode == 'image_rna'{RESET}{CARRIBEAN_GREEN}. Will load TRAINING indices used during the last unimodal training run{RESET}"         )
                
          if args.input_mode == 'image':
            fqn = f"{args.data_dir}/train_inds_image"
            if DEBUG>6:
              print ( f"LOADER:         INFO:     about to load train_inds from = {MAGENTA}{fqn}{RESET}"         )
            with open(fqn, 'rb') as f:
              test_inds = pickle.load(f)
              if DEBUG>6:
                  print ( f"LOADER:         INFO:     test_inds              = {PINK}{test_inds}{RESET}"         )
                  
          elif args.input_mode == 'rna':
            fqn = f"{args.data_dir}/train_inds_rna"
            if DEBUG>6:
              print ( f"LOADER:         INFO:     about to load train_inds  from = {MAGENTA}{fqn}{RESET}"         )
            with open(fqn, 'rb') as f:
              test_inds = pickle.load(f)
              if DEBUG>6:
                  print ( f"LOADER:         INFO:     test_inds.type         = {BLEU}{type(test_inds)}{RESET}"    )
                  print ( f"LOADER:         INFO:     test_inds              = {BLEU}{test_inds}{RESET}"          )




    # 4  Determine number of mini-batches required (and sanity check)
    
    if just_test!='True':
      
      if DEBUG>2:
          print ( f"LOADER:         INFO:     len(train_inds)            = {BLEU}{len(train_inds)}{RESET}"        )
          print ( f"LOADER:         INFO:     len(test_inds)             = {BLEU}{len(test_inds) }{RESET}"        )    
                
      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                                        train   test"               )
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                      mini-batch size: {MIKADO}{batch_size:>6d}, {batch_size:>5d}{RESET}"               )
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                             for {MIKADO}{pct_test*100:>3.0f}%{RESET} split,  samples: {MIKADO}{len(train_inds):>6d}, {len(test_inds):>5d}{RESET}" )
        if args.input_mode == 'image':
          print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                              cases:   {MIKADO}{int(len(train_inds)/n_tiles[0]):>6d}, {int(len(test_inds)/n_tiles[0]):>5d}{RESET}" )

      number_of_train_batches = len(train_inds)//batch_size
      number_of_test_batches  = len(test_inds) //batch_size
        
      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                              batches: {MIKADO}{number_of_train_batches:>6d}, {number_of_test_batches:>5d}{RESET}" )
    
    else:

      if DEBUG>0:
          print ( f"LOADER:         INFO:     len(test_inds)             = {BLEU}{len(test_inds) }{RESET}"         )  

      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                                       test"               )
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                      mini-batch size: {MIKADO}{batch_size:>5d}{RESET}"               )
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                               for {MIKADO}{pct_test*100:>3.0f}%{RESET}  examples:     {MIKADO}{len(test_inds):>5d}{RESET}" )
        if args.input_mode == 'image':
          print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                              cases:   {MIKADO}{int(len(test_inds)/n_tiles[0]):>5d}{RESET}" )

      number_of_test_batches  = len(test_inds)//batch_size

      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                          batches: {MIKADO}{number_of_test_batches:>5d}{RESET}" )
    

    if just_test!='True':
      if number_of_train_batches<1:
        print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TRAINING batches (consider re-running the tiler or reducing 'BATCH_SIZE' (currently {CYAN}{batch_size}{RESET}{RED}) or REDUCING 'PCT_TEST' (currently {CYAN}{pct_test}{RESET}){RED} ) -- halting now{RESET}")
        sys.exit(0)

    if number_of_test_batches<1:
        print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TEST batches (consider re-running the tiler or reducing 'BATCH_SIZE' (currently {CYAN}{batch_size}{RESET}{RED}) or REDUCING 'PCT_TEST' (currently {CYAN}{pct_test}{RESET}){RED} ) -- halting now{RESET}")
        sys.exit(0)



    # 5 create and return the various train and test loaders
    
    
    # If data set size is indivisible by batch size, drop last incomplete batch. Dropping the last batch is fine because we randomly subsample from the data set, meaning all data should be sampled uniformly in expectation.
    #
    DROP_LAST = True
    #
    # This is Soumith's recommended approach. See:
    #
    #     https://github.com/pytorch/pytorch/issues/1106
    #
    
    
    
    # 5A train_loader (two cases: single CPU and distributed)
    
    if just_test!='True':
      
      # 5A  train loader
  
      if args.ddp=='False':   # Single GPU <-- Main case

        if DEBUG>2:
          print( "LOADER:         INFO:   374: about to create and return train loader - single GPU case" )
                
        train_loader   = DataLoader(
          dataset,
          batch_size   = batch_size,
          num_workers  = num_workers,
          sampler      = SubsetRandomSampler( train_inds ),
          drop_last    = DROP_LAST,
          pin_memory   = pin_memory                                                                        # Move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
          )        
      
      else:                                                                                                # Multiple GPUs. DistributedSampler will handle dispensing batches to GPUs

        if DEBUG>2:
          print( "LOADER:         INFO:   about to create and return train loader - multiple GPU / distributed sampler case" )
                
        if DEBUG>2:
          print ( f"{BRIGHT_GREEN}LOADER:         INFO:     DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! about to initialize DistributedSampler:{RESET}" )
          print ( f"LOADER:         INFO:       world_size          = {MIKADO}{world_size}{RESET}"          ) 
          print ( f"LOADER:         INFO:       rank                = {MIKADO}{rank}{RESET}"                )
          print ( f"LOADER:         INFO:       num_workers         = {MIKADO}{num_workers}{RESET}"         )

        train_loader = DataLoader(
          dataset, 
          batch_size   = batch_size,
          num_workers  = 0,
          shuffle      = False,
          sampler      = torch.utils.data.distributed.DistributedSampler( dataset, num_replicas = world_size, rank = rank ) # makes sure each process gets a different slice of the training data
          )
  
  
      # 5B test_loader for the *training* phase: i.e. ./do_all -d stad -i image. We already have a loader for the training indices; here we define a loader for the test indices: testing during the training phase

      if DEBUG>2:
        print( "LOADER:         INFO:   408: about to create and return test  loader (the one that's used in the training phase after each epoch for validation testing)" )

      if args.ddp=='False':   # Single GPU <-- Main case

        if DEBUG>2:
          print( "LOADER:         INFO:   413:   single GPU case" ) 
    
        test_loader = DataLoader(
          dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset if input_mode=='rna' else dataset_image_test,
          batch_size   = batch_size,
          num_workers  = num_workers,
          sampler      = SubsetRandomSampler( test_inds ),              
          drop_last    = DROP_LAST,
          pin_memory   = pin_memory                                                                        # Move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
          )
          
      
      
      else:

        if DEBUG>2:
          print( "LOADER:         INFO:   about to create and return test loader for training mode -  multiple GPU / distributed sampler case" ) 
          
        num_workers    = 0
        sampler        = torch.utils.data.distributed.DistributedSampler(                                  # makes sure that each process gets a different slice of the testing data
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
          dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset_image_test,
          batch_size       = batch_size,
          num_workers      = num_workers,
          shuffle          = False,
          sampler          = sampler
          )   
   
       
    # 5C test_loader for the DEDICATED test mode: i.e. ./just_test -d stad -i image  (NOT testing during the training phase)
 
    else:   # just_test=='True'

      if DEBUG>2:
        print( "LOADER:         INFO:   about to create and return test loader for the dedicated test mode: ('just_test'). Note: sequential rather than random sampling" ) 

      if DEBUG>2:
        print( f"LOADER:         INFO:        args.cases  = {AMETHYST}{args.cases}{RESET}"       )
        print( f"LOADER:         INFO:         test_inds  = {AMETHYST}{test_inds}{RESET}"        )
        print( f"LOADER:         INFO:        batch_size  = {AMETHYST}{batch_size}{RESET}"       )

      dataset = dataset if args.input_mode=='rna' else dataset_image_test if args.cases!='ALL_ELIGIBLE_CASES' else dataset
      
      if use_autoencoder_output=='True':

        sampler     = SubsetRandomSampler( test_inds )                                                     # tiles need to be drawn at random because we want at many different parts of the image represented in the autoencoder output
      else:
        sampler     = SequentialSampler  ( test_inds )                                                     # tiles need to be drawn sequentially because we are analysing a 2D contiguous square patch of tiles 
      
      test_loader = DataLoader(
        dataset,
        sampler     = sampler,
        batch_size  = batch_size,
        num_workers = 1,
        drop_last   = DROP_LAST,
        pin_memory  = pin_memory
        )
        


    if args.input_mode=='image':
      final_batch_size =   (final_test_batch_size*batch_size) if (final_test_batch_size*batch_size)<len(test_inds) else batch_size
    elif args.input_mode=='rna':
      final_batch_size  =  len(test_inds)
    elif args.input_mode=='image_rna':
      final_batch_size  =  len(test_inds)


    num_workers            =  num_workers
    final_test_loader = DataLoader(
      dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset if input_mode=='rna' else dataset_image_test,
      batch_size  = final_batch_size,
      num_workers = num_workers,
      sampler     = SubsetRandomSampler( test_inds ),
      drop_last   = DROP_LAST,
      pin_memory  = pin_memory                                                                             # move loaded and processed tensors into CUDA pinned memory. See: http://pytorch.org/docs/master/notes/cuda.html
      )

    if DEBUG>99:    
      print( f"LOADER:         INFO:   test_loader  = {PURPLE}{test_loader}{RESET}" )
    
    torch.cuda.empty_cache()
    
    if just_test!='True':
      return train_loader, test_loader, final_batch_size, final_test_loader
    else:
      return 0,            test_loader, final_batch_size, final_test_loader
      
      
