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

from modes import classifyConfig
from modes import GTExV6Config
from modes import MnistConfig
from modes import pre_compressConfig

from constants  import *

DEBUG=1


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_config( dataset, lr, batch_size ):
  
    """
    Return configuration object based on dataset string.
    """

    SUPPORTED_MODES = [ 'gtexv6', 'classify', 'pre_compress', 'analyse_data', 'mnist']
    
    if dataset not in SUPPORTED_MODES:
        raise ValueError('Dataset %s is not supported.' % dataset)

    if   dataset == 'gtexv6':
        return GTExV6Config( )
    elif dataset == 'classify':
        return classifyConfig     ( lr,  batch_size )
    elif dataset == 'pre_compress':
        return pre_compressConfig ( lr,  batch_size )
    elif dataset == 'analyse_data':
        return pre_compressConfig ( lr,  batch_size )                                                      # uses the pre_compress() config file
    elif dataset == 'mnist':
        return MnistConfig()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_data_loaders( args, gpu, cfg, world_size, rank, batch_size, n_samples, num_workers, pin_memory, pct_test, writer, directory=None) :

      
    """
    Create and return dataset(s) and data loaders for train and test datasets as appropriate
    """
    
    input_mode             = args.input_mode
    n_tiles                = args.n_tiles
    final_test_batch_size  = args.final_test_batch_size
    just_test              = args.just_test
    use_autoencoder_output = args.use_autoencoder_output
    mode                   = args.mode
    nn_type_img            = args.nn_type_img
    nn_type_rna            = args.nn_type_rna

    we_are_autoencoding=False
    if ( (input_mode=='image') &  ('AE' in nn_type_img[0]) )  |  ( (input_mode=='rna') & ('AE' in nn_type_rna[0]) ):
      we_are_autoencoding=True
      
    
    # 1 Preparation

    if just_test=='True':
      pct_test=1.0                                                                                         # In test mode, all tiles are test tiles, by definition.  Let's make sure.
      


    # 2 Fetch dataset(s)
    
    if DEBUG>100:
      print( f"{RESET}LOADER:         INFO:    about to load applicable dataset(s)" )

    if input_mode=='image':

      if args.cases!='ALL_ELIGIBLE_CASES':   # catering for OTHER than 'ALL_ELIGIBLE_CASES'. There are separate database files for training and testing. 
        
        # always load the test dataset ... (and if we are in just_test mode, that's all we need)
        which_dataset      = 'dataset_image_test'      
        dataset_image_test = cfg.get_dataset( args, which_dataset, gpu )
        # equates via cfg.get_dataset to: dataset = classifyDataset( cfg, which_dataset, args ), i.e. make an object of class classifyDataset using it's __init__() constructor
        # and dataset_image_test.images = data_image_test['images'] etc.; noting that 'data_image_test' is a tensor: see dataset() where data = torch.load(f"data/classify/{which_dataset}.pth"
        
        if DEBUG>100:    
          print( f"LOADER:         INFO:        dataset {CYAN}{which_dataset}{RESET} now loaded" )      
  
        test_inds = list(range(len( dataset_image_test )  )   )

        if DEBUG>80:
          print( f"LOADER:         INFO:    test_inds  = \n{MIKADO}{test_inds}{RESET}" ) 
                  
        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled
          random.shuffle( test_inds )
        
        
        # ... but load the training dataset only if we're in training mode
        if just_test!='True':
            
          which_dataset = 'dataset_image_train'
          dataset       = cfg.get_dataset( args, which_dataset, gpu )
    
          if DEBUG>2:    
            print( f"LOADER:         INFO:        dataset {CYAN}{which_dataset}{RESET} now loaded" )     
                
          train_inds = list(range(len( dataset )  )   )  

          if DEBUG>2:
            print( f"LOADER:         INFO:    train_inds  = \n{MIKADO}{train_inds}{RESET}"                 )
            
          random.shuffle(train_inds)                                                                       # in training mode, it's critical that both the training and test sets are shuffled
            

      else:     # catering for ALL_ELIGIBLE_CASES.  Different _indices_ (but all drawn from dataset_image_train) are used for training and testing. Not very useful for image training, since training and test tiles could derive from the same slides.

        which_dataset = 'dataset_image_train'      
        # ~ dataset       = cfg.get_dataset( args, which_dataset, writer, gpu )                            # 21-09-23  removed introduced error caused by the presence of 'writer' parameter
        dataset       = cfg.get_dataset( args, which_dataset, gpu )
        
        if DEBUG>8:    
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

        else:                                                                                              # test mode / ALL_ELIGIBLE_CASES

          if use_autoencoder_output!='True':                                                               # Not using the autoencoder is the default case (unimode 'just_test' to create patches)

            split      = math.floor(len(indices) * (1 - pct_test))                                   
            train_inds = indices[:split]
            test_inds  = indices[split:]
  
            if DEBUG>44:
              print( f"LOADER:         INFO:    train_inds                  = \n{CARRIBEAN_GREEN}{train_inds}{RESET}" )
              print( f"LOADER:         INFO:    test_inds                   = \n{CARRIBEAN_GREEN}{test_inds}{RESET}"  )
       
          else:                                                                                            # autoencoding as a prelude to clustering 

            test_inds  = indices                                                                           # when using an autoencoder, we want to be able to process every tile in test mode, in particular so that we have as many tiles as possible to use when clustering

            if DEBUG>44:
              print( f"LOADER:         INFO:    test_inds                   = \n{BITTER_SWEET}{test_inds}{RESET}"  )                                                                                      
            

    elif input_mode=='rna':

      if args.cases!='ALL_ELIGIBLE_CASES':                                                                 # catering for OTHER than 'ALL_ELIGIBLE_CASES'. There are separate database files for training and testing. 
      
        # always load the test dataset ... (and if we are in just_test mode, that's all we need)
        dataset_rna_test       = cfg.get_dataset( args, 'dataset_rna_test',       gpu )
        test_inds = list(range(len( dataset_rna_test )  )   )
        
        if DEBUG>10:
          print( f"LOADER:         INFO:    test_inds  = \n{MIKADO}{test_inds}{RESET}" ) 
                  
        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled
          random.shuffle( test_inds )
        
        
        # ... load the training dataset as well IFF we're in training mode
        if just_test!='True':
          dataset    = cfg.get_dataset( args, 'dataset_rna_train',        gpu )
          train_inds = list(range(len( dataset )  )   )

          if DEBUG>10:
            print( f"LOADER:         INFO:    train_inds  = \n{MIKADO}{train_inds}{RESET}"                 )
            
          random.shuffle(train_inds)                                                                       # in training mode, it's critical that both the training and test sets are shuffled
            
      

      else:     # catering for ALL_ELIGIBLE_CASES.  Different indices (but always drawn from dataset_rna_train) for training and testing.

        dataset = cfg.get_dataset( args, 'dataset_rna_train',            gpu )
        indices = list(range( len( dataset)  )   )
        
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

          if use_autoencoder_output!='True':                                                               # Not using the autoencoder is the default case (unimode 'just_test' to create patches)

            split      = math.floor(len(indices) * (1 - pct_test))                                   
            train_inds = indices[:split]
            test_inds  = indices[split:]
  
            if DEBUG>44:
              print( f"LOADER:         INFO:    train_inds                  = \n{CARRIBEAN_GREEN}{train_inds}{RESET}" )
              print( f"LOADER:         INFO:    test_inds                   = \n{CARRIBEAN_GREEN}{test_inds}{RESET}"  )
       
          else:                                                                                            # autoencoding as a prelude to clustering 

            test_inds  = indices                                                                           # when using an autoencoder, we want to be able to process every example in test mode, in particular so that we have as many examples as possible to use when clustering
            if DEBUG>44:
              print( f"LOADER:         INFO:    test_inds                   = \n{BITTER_SWEET}{test_inds}{RESET}"  )                                                                                      


    elif input_mode=='image_rna':

      if args.cases!='ALL_ELIGIBLE_CASES':                                                                 # catering for OTHER than 'ALL_ELIGIBLE_CASES'. There are separate database files for training and testing. 
      
        # always load the test dataset ... (and if we are in just_test mode, that's all we need)
        dataset_image_rna_test = cfg.get_dataset( args, 'dataset_image_rna_test', gpu )
        test_inds = list(range(len( dataset_image_rna_test )  )   )
        
        if DEBUG>0:
          print( f"LOADER:         INFO:    test_inds  = \n{MIKADO}{test_inds}{RESET}" ) 
                  
        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled
          random.shuffle( test_inds )
        
        
        # ... load the training dataset as well IFF we're in training mode
          dataset    = cfg.get_dataset( args, 'dataset_image_rna_train',  gpu )
          train_inds = list(range(len( dataset )  )   )

          if DEBUG>0:
            print( f"LOADER:         INFO:    train_inds  = \n{MIKADO}{train_inds}{RESET}"                 )
            
          random.shuffle(train_inds)                                                                       # in training mode, it's critical that both the training and test sets are shuffled
            
      

      else:     # catering for ALL_ELIGIBLE_CASES.  Different indices (but always drawn from dataset_rna_train) for training and testing.

        dataset = cfg.get_dataset( args, 'dataset_image_rna_train',      gpu )
        indices = list(range( len( dataset)  )   )
        


        if DEBUG>44:
          print( f"LOADER:         INFO:    indices                         = \n{MIKADO}{indices}{RESET}"   )

        if just_test!='True':                                                                              # in training mode, it's critical that both the training and test sets are shuffled ...

          random.shuffle( indices )                                                                        # ... (in test mode, we only use the test indices, and they must not be shuffled as we have to recreate the patches for visualization on Tensorboard)
           
          split      = math.floor(len(indices) * (1 - pct_test))                                   
          train_inds = indices[:split]
          test_inds  = indices[split:]
          
          if DEBUG>44:
            print( f"LOADER:         INFO:    train_inds  ( after shuffle ) = \n{MIKADO}{train_inds}{RESET}" )
            print( f"LOADER:         INFO:    test_inds   ( after shuffle ) = \n{MIKADO}{test_inds}{RESET}"  )

        else:

          if use_autoencoder_output!='True':                                                               # Not using the autoencoder is the default case (unimode 'just_test' to create patches)

            split      = math.floor(len(indices) * (1 - pct_test))                                   
            train_inds = indices[:split]
            test_inds  = indices[split:]
  
            if DEBUG>44:
              print( f"LOADER:         INFO:    train_inds                  = \n{CARRIBEAN_GREEN}{train_inds}{RESET}" )
              print( f"LOADER:         INFO:    test_inds                   = \n{CARRIBEAN_GREEN}{test_inds}{RESET}"  )
       
          else:                                                                                            # autoencoding as a prelude to clustering 

            test_inds  = indices                                                                           # when using an autoencoder, we want to be able to process every example in test mode, in particular so that we have as many examples as possible to use when clustering
            if DEBUG>44:
              print( f"LOADER:         INFO:    test_inds                   = \n{BITTER_SWEET}{test_inds}{RESET}"  )                                                                                      



    
  

    # 3A maybe save indices used during training for later use in 'image_rna' test mode (see comment 3B for an explanation)
    
    if args.cases=='UNIMODE_CASE____MATCHED':
    
      if just_test!='True':                                                                                # training mode
  
        #  3A save training indices for possible later use in test
        
        if args.input_mode == 'image':
          if DEBUG>99:
            print ( f"LOADER:         INFO:     (unmodified) train_inds              = {PINK}{train_inds}{RESET}"               )
          fqn = f"{args.data_dir}/train_inds_image"
          if DEBUG>99:
                print ( f"LOADER:         INFO:     about to save train_inds to = {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"      )
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
            print ( f"LOADER:         INFO:     about to save train_inds to: {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"          )
          with open(fqn, 'wb') as f:
            pickle.dump(train_inds, f)
  
          if DEBUG>99:
            print ( f"LOADER:         INFO:     (unmodified) test_inds              = {BLEU}{test_inds}{RESET}"               )
          fqn = f"{args.data_dir}/test_inds_rna"
          if DEBUG>99:
            print ( f"LOADER:         INFO:     about to save test_inds  to: {MAGENTA}{fqn}{RESET} for later use in {CYAN}test{RESET} mode ({CYAN}just_test=='True'{RESET})"          )
          with open(fqn, 'wb') as f:
            pickle.dump(test_inds, f)

  
      # 3B If the multimode flag is set, then, for 'image' TEST mode and 'rna' TEST mode (but NOT 'image_rna' TEST mode) retrieve and use the TRAINING indices that were used during unimodal training.  
      #    We want to generate as many feature vectors as possible to train the image+rna model
              
      elif just_test=='True':                                                                              # test mode     
        
        if args.multimode == 'image_rna':
        
          if DEBUG>33:
              print ( f"{CARRIBEAN_GREEN}LOADER:         NOTE:     {MAGENTA}args.just_test == 'True'{RESET}{CARRIBEAN_GREEN} and {MAGENTA}args.multimode == 'image_rna'{RESET}{CARRIBEAN_GREEN}. Will load TRAINING indices used during the last unimodal training run{RESET}"         )
                
          if args.input_mode == 'image':
            fqn = f"{args.data_dir}/train_inds_image"
            if DEBUG>6:
              print ( f"LOADER:         INFO:     about to load train_inds from = {MAGENTA}{fqn}{RESET}"          )
            with open(fqn, 'rb') as f:
              test_inds = pickle.load(f)
              if DEBUG>6:
                  print ( f"LOADER:         INFO:     test_inds              = {PINK}{test_inds}{RESET}"          )
                  
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
      
      if DEBUG>0:
        train_cases = len(train_inds)
        test_cases  = len(test_inds)
        
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                     train   test"               )
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                          for {MIKADO}{pct_test*100:>3.0f}%{RESET} split,  samples: {MIKADO}{train_cases:>6d}, {test_cases:>5d}  {DULL_WHITE} <<< note: samples used won't always equal {CYAN}N_SAMPLES{RESET}{DULL_WHITE} because of quantisation introduced by mini-batches, which must always be full (residual discarded){RESET}" )
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                   mini-batch size: {MIKADO}{batch_size:>6d}, {batch_size:>5d}{RESET}"               )

      number_of_train_batches = train_cases//batch_size
      number_of_test_batches  = test_cases //batch_size
        
      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                           batches: {MIKADO}{number_of_train_batches:>6d}, {number_of_test_batches:>5d}{RESET}" )
    
    else:

      if DEBUG>0:
          print ( f"LOADER:         INFO: (just_test) len(test_inds)             = {BLEU}{len(test_inds) }{RESET}"         )  

      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                                       test"                                                                   , flush=True )
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                      mini-batch size: {MIKADO}{batch_size:>5d}{RESET}"                                        , flush=True )
        print( f"{CLEAR_LINE}LOADER:         INFO: (just_test)                                                                                   for {MIKADO}{pct_test*100:>3.0f}%{RESET}  examples: {MIKADO}{len(test_inds):>5d}{RESET}"    , flush=True )

      number_of_test_batches  = len(test_inds)//batch_size

      if DEBUG>0:
        print( f"{CLEAR_LINE}LOADER:         INFO:                                                                                                          batches: {MIKADO}{number_of_test_batches:>5d}{RESET}" )
    

    if just_test!='True':
      if number_of_train_batches<1:
        print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TRAINING batches (consider re-running the tiler or reducing 'BATCH_SIZE' (currently {CYAN}{batch_size}{RESET}{RED}) or REDUCING 'PCT_TEST' (currently {CYAN}{pct_test}{RESET}{RED} ) -- halting now{RESET}")
        sys.exit(0)

    if number_of_test_batches<1:
        if args.input_mode == 'image':      
          print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TEST batches. Consider re-running the tiler or try REDUCING the {CYAN}BATCH_SIZE ('-1'){RESET}{RED} (currently {MIKADO}{batch_size}{RESET}{RED}) to not more than {MIKADO}{len(test_inds)}{RESET}{RED} or REDUCING {CYAN}PCT_TEST ('-1') {RESET}{RED}(currently {MIKADO}{pct_test}{RESET}{RED}){RESET}")
        else:
          print( f"{RED}LOADER:         FATAL: The combination of the chosen {CYAN}BATCH_SIZE{RESET}{RED} and {CYAN}N_TILES{RESET}{RED} would result in there being zero TEST batches.")
          print( f"{RED}LOADER:         FATAL: Possible remedy 1: Try REDUCING the {CYAN}BATCH_SIZE ('-b'){RESET}{RED} (currently {MIKADO}{batch_size}{RESET}{RED}) to not more than {MIKADO}{len(test_inds)}{RESET}{RED}")
          print( f"{RED}LOADER:         FATAL: Possible remedy 2: Try INCREASING {CYAN}PCT_TEST ('-1') {RESET}{RED}(currently {MIKADO}{pct_test}{RESET}{RED}){RESET}")
          print( f"{RED}LOADER:         FATAL: Possible remedy 3: Try regenerating the dataset {CYAN}REGENERATE ('-r True') {RESET}{RED}){RESET}")
        print( f"{RED}LOADER:         FATAL: can't continue -- halting now{RESET}")
        sys.exit(0)


    if just_test!='True':
      total_batches  = number_of_train_batches + number_of_test_batches
    else:
      total_batches  = number_of_test_batches


    # ~ if percent_unused > 2.5:
    if DEBUG>0:
      if input_mode=='image':
        if just_test != 'True':
          used_train_tiles         = number_of_train_batches * batch_size
          used_test_tiles          = number_of_test_batches  * batch_size
          percent_train_tiles_used = 100 * used_train_tiles / train_cases 
          percent_test_tiles_used  = 100 * used_test_tiles  / test_cases 
          print( f"{ORANGE}CLASSI:         INFO:   available training tiles \r\033[60C = {MIKADO}{train_cases:>5d}{RESET}{ORANGE}.  Training tiles that will be used{RESET}{ORANGE} = training  batches ({MIKADO}{number_of_train_batches:>4d}{RESET}{ORANGE}) * batch_size ({MIKADO}{batch_size:>3d}{RESET}{ORANGE}) = {MIKADO}{used_train_tiles}{RESET}{ORANGE}.  Hence {MIKADO}{used_train_tiles:>5d}{RESET}{ORANGE}/{MIKADO}{train_cases:>5d}{RESET}{ORANGE} ~ {BRIGHT_GREEN if percent_train_tiles_used >98 else RED if percent_train_tiles_used <95 else BOLD_ORANGE}{100*used_train_tiles/train_cases:3.0f}%{RESET}{ORANGE} of the available training tiles will be used in           training{RESET}{ORANGE}. Change {CYAN}BATCH_SIZE{RESET}{ORANGE} or {CYAN}PCT_TEST{RESET}{ORANGE} if this is unacceptable{RESET}")
          print( f"{ORANGE}CLASSI:         INFO:   available test     tiles \r\033[60C = {MIKADO}{ test_cases:>5d}{RESET}{ORANGE}.  Test     tiles that will be used{RESET}{ORANGE} = test      batches ({MIKADO}{ number_of_test_batches:>4d}{RESET}{ORANGE}) * batch_size ({MIKADO}{batch_size:>3d}{RESET}{ORANGE}) = {MIKADO}{ used_test_tiles}{RESET}{ORANGE}.  Hence {MIKADO}{used_test_tiles :>5d}{RESET}{ORANGE}/{MIKADO}{ test_cases:>5d}{RESET}{ORANGE} ~ {BRIGHT_GREEN if percent_test_tiles_used  >98 else RED if percent_test_tiles_used  <95 else BOLD_ORANGE}{100*used_test_tiles / test_cases:3.0f}%{RESET}{ORANGE} of the available test     tiles will be used in per-epoch testing{RESET}{ORANGE}.  Change {CYAN}BATCH_SIZE{RESET}{ORANGE} or {CYAN}PCT_TEST{RESET}{ORANGE} if this is unacceptable{RESET}")
          print( "\n" )
      else:
        used_samples   = total_batches*batch_size
        unused_samples = n_samples - used_samples 
        percent_unused = 100*unused_samples/n_samples
        print( f"{ORANGE}CLASSI:         INFO:   {CYAN}N_SAMPLES{RESET}{ORANGE} is {MIKADO}{n_samples}{RESET}{ORANGE}, total batches = {MIKADO}{total_batches}{RESET}{ORANGE} and {CYAN}BATCH_SIZE{RESET}{ORANGE} = {MIKADO}{batch_size}{RESET}{ORANGE}. This means {MIKADO}{used_samples}{RESET}{ORANGE} out of the {MIKADO}{n_samples}{RESET}{ORANGE} available samples ({MIKADO}{100-percent_unused:.0f}%{RESET}{ORANGE}) are being used in this training run{RESET}{ORANGE}. Change {CYAN}BATCH_SIZE{RESET}{ORANGE} and {CYAN}PCT_TEST{RESET}{ORANGE} if this figure is unacceptable{RESET}")
        print( "\n" )




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
          print( "LOADER:         INFO:   about to create and return train loader - single GPU case" )
                
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
  
  
      # 5B test_loader for the *training* phase: i.e. ./do_all -d stad -i image. We already have a loader for the training indices; here we define a loader for the test indices: that is, testing during the training phase

      if DEBUG>2:
        print( "LOADER:         INFO:   408: about to create and return test  loader (the one that's used in the training phase after each epoch for validation testing)" )

      if args.ddp=='False':   # Single GPU <-- Main case

        if DEBUG>2:
          print( "LOADER:         INFO:   413:   single GPU case" ) 
    
        test_loader = DataLoader(
          dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset_rna_test if input_mode=='rna' else dataset_image_rna_test if input_mode=='image_rna' else dataset_image_test,
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
          dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset_rna_test if input_mode=='rna' else dataset_image_rna_test if input_mode=='image_rna'  else dataset_image_test,
          batch_size       = batch_size,
          num_workers      = num_workers,
          shuffle          = False,
          sampler          = sampler
          )   
   
       
    # 5C test_loader for the DEDICATED test mode: i.e. ./just_test -d stad -i xxxx  (NOT testing during the training phase)
 
    else:   # just_test=='True'

      if DEBUG>2:
        print( f"LOADER:         INFO:        args.cases  = {AMETHYST}{args.cases}{RESET}"       )
        print( f"LOADER:         INFO:         test_inds  = {AMETHYST}{test_inds}{RESET}"        )
        print( f"LOADER:         INFO:        batch_size  = {AMETHYST}{batch_size}{RESET}"       )

      if args.input_mode=='image':
        if DEBUG>0:
          print( f"{ORANGE}LOADER:         INFO:   about to create and return test loader for the dedicated test mode: (image / 'just_test'). Note: sequential rather than random sampling for test mode{RESET}" ) 
        dataset     = dataset_image_test      
        sampler     = SequentialSampler  ( test_inds )                                                     # tiles need to be drawn sequentially because we are analysing a 2D contiguous square patch of tiles 
      elif args.input_mode=='rna':
        if we_are_autoencoding:                                                                                                                          
          if DEBUG>0:
            print( f"{BOLD}{CHARTREUSE}LOADER:         INFO:   about to create and return loader for the dedicated test mode{RESET}" )
            print( f"{BOLD}{CHARTREUSE}LOADER:         NOTE:   autoencoder working is enabled {RESET}" )
          which_dataset = 'dataset_rna_test'
          dataset_rna_test = cfg.get_dataset( args, which_dataset, gpu )
          dataset = dataset_rna_test
          sampler = SubsetRandomSampler( test_inds )
        else:
          if DEBUG>0:
            print( "LOADER:         INFO:   about to create and return test loader for the dedicated test mode: (rna / 'just_test')" ) 
          dataset = dataset_rna_test
          sampler     = SubsetRandomSampler( test_inds ) 
      elif args.input_mode=='image_rna':
        if DEBUG>0:
          print( "LOADER:         INFO:   about to create and return test loader for the dedicated test mode: (image_rna / 'just_test')" ) 
        dataset = dataset_image_rna_test
        sampler     = SubsetRandomSampler( test_inds )   


      if use_autoencoder_output=='True':
        sampler     = SubsetRandomSampler( test_inds )                                                     # for autoencoder output (only), tiles need to be drawn at random because we want as many different parts of the image as possible represented in the autoencoder output
      else:
        sampler     = SequentialSampler  ( test_inds )
        
      
      test_loader = DataLoader(
        dataset,
        sampler     = sampler,
        batch_size  = batch_size,
        num_workers = 1,
        drop_last   = DROP_LAST,
        pin_memory  = pin_memory
        )
        


    if args.input_mode   == 'image':
      final_batch_size =   (final_test_batch_size*batch_size) if (final_test_batch_size*batch_size)<len(test_inds) else batch_size
    elif args.input_mode == 'rna':
      final_batch_size  =  len(test_inds)
    elif args.input_mode == 'image_rna':
      final_batch_size  =  len(test_inds)


    final_test_loader = DataLoader(
      dataset if args.cases=='ALL_ELIGIBLE_CASES' else dataset_rna_test if input_mode=='rna' else dataset_image_rna_test if input_mode=='image_rna'  else dataset_image_test,
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
      
      
