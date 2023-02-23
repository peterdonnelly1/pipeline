"""================================================================================================================================================

Generate torch dictionary from pre-processed TCGA image tiles or gene expression vectors
            
================================================================================================================================================"""

import cv2
import os
import re
import sys
import time
import cuda
import cupy
import torch
import shutil
import random
import fnmatch
import numpy as np
import pandas as pd
from pathlib import Path


from modes.pre_compress.config import pre_compressConfig

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

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
DULL_BLEU='\033[38;2;0;102;204m'
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

SUCCESS=1
DEBUG=1

def generate( args, class_names, n_samples, highest_class_number, multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count, pct_test, n_tiles, tile_size, gene_data_norm, gene_data_transform ):

  # DON'T USE args.n_samples or args.n_tiles or args.gene_data_norm or args.tile_size or args.highest_class_number since these are job-level lists. Here we are just using one value of each, passed in as the parameters above
  data_dir                     = args.data_dir
  input_mode                   = args.input_mode
  pretrain                     = args.pretrain
  cases                        = args.cases
  cases_reserved_for_image_rna = args.cases_reserved_for_image_rna
  rna_file_name                = args.rna_file_name
  tcga_rna_seq_file_suffix              = args.tcga_rna_seq_file_suffix  
  rna_file_reduced_suffix      = args.rna_file_reduced_suffix
  class_numpy_file_name        = args.class_numpy_file_name
  use_autoencoder_output       = args.use_autoencoder_output
  use_unfiltered_data          = args.use_unfiltered_data 


  class_counts = np.zeros( highest_class_number+1, dtype=np.int )

  if DEBUG>6:
    print( "P_C_GENERATE:   INFO:   \
   input_mode=\033[36;1m{:}\033[m,\
   data_dir=\033[36;1m{:}\033[m,\
   n_samples=\033[36;1m{:}\033[m,\
   n_tiles=\033[36;1m{:}\033[m,\
   tile_size=\033[36;1m{:}\033[m,\
   gene_data_norm=\033[36;1m{:}\033[m,\
   gene_data_transform=\033[36;1m{:}\033[m,\
   rna_file_name=\033[36;1m{:}\033[m,\
   class_numpy_file_name=\033[36;1m{:}\033[m,\
   n_tiles=\033[36;1m{:}\033[m"\
  .format( input_mode, data_dir, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform, rna_file_name, class_numpy_file_name, n_tiles), flush=True )
 
  cfg = pre_compressConfig( 0,0 )


  # (1) analyse data directory

  if use_unfiltered_data==True:
    rna_suffix = tcga_rna_seq_file_suffix[1:]
  else:
    rna_suffix = rna_file_reduced_suffix

  cumulative_image_file_count = 0
  cumulative_png_file_count   = 0
  cumulative_rna_file_count   = 0
  cumulative_other_file_count = 0
  
  for dir_path, dirs, files in os.walk( data_dir ):                                                        # each iteration takes us to a new directory under data_dir

    if not (dir_path==data_dir):                                                                           # the top level directory is skipped because it only contains sub-directories, not data      
      
      image_file_count   = 0
      rna_file_count     = 0
      png_file_count     = 0
      other_file_count   = 0

      for f in sorted( files ):
       
        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
          image_file_count            +=1
          cumulative_image_file_count +=1
        elif  ( f.endswith( 'png' ) ):
          png_file_count              +=1
          cumulative_png_file_count   +=1
        elif  ( f.endswith( rna_suffix ) ):
          rna_file_count              +=1
          cumulative_rna_file_count   +=1
        else:
          other_file_count            +=1
          cumulative_other_file_count +=1
        
      if DEBUG>77:
        if ( ( rna_file_count>1 ) | ( image_file_count>1 ) ): 
          print( f"P_C_GENERATE:   INFO:    \033[58Cdirectory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq file(s) and {MIKADO}{image_file_count:<2d}{RESET} image files(s) and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )
          time.sleep(0.5)       
        else:
          print( f"P_C_GENERATE:   INFO:    directory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq files, {MIKADO}{image_file_count:<2d}{RESET} image files and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )






  # (2) process IMAGE data if applicable
  
  if ( input_mode=='image' ) & ( pretrain!='True' ):


    # check to see that there actually are tiles to process
     
    if cumulative_png_file_count==0:
      print ( f"{RED}P_C_GENERATE:    FATAL:  there are no tile files ('png' files) at all. To generate tiles, run '{CYAN}./do_all.sh -d <cancer type code> -i image -c <CASES SELECTOR>{RESET}{RED}' ... halting now{RESET}", flush=True )                 
      sys.exit(0)         
  
    print( f"{ORANGE}P_C_GENERATE:   INFO:    input_mode is '{RESET}{CYAN}{input_mode}{RESET}{ORANGE}', so rna and other data will not be generated{RESET}" )  
  
      
    if args.just_test=='True':

      #  (2A) generate Test dataset

      if args.cases == 'UNIMODE_CASE____IMAGE_TEST':

        target                = 'image_test'
        cases_required        = n_samples
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}P_C_GENERATE:   INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) n_tiles (this run)................................................................. = {MIKADO}{n_tiles}{RESET}",                flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) cases_required .................................................................... = {MIKADO}{cases_required}{RESET}",         flush=True )
  
        global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts )

        if DEBUG>0:
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )


      elif args.cases == 'MULTIMODE____TEST':

        target                = 'image_test'
        cases_required        = cases_reserved_for_image_rna
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}P_C_GENERATE:   INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) n_tiles (this run)................................................................. = {MIKADO}{n_tiles}{RESET}",                flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) cases_required .................................................................... = {MIKADO}{cases_required}{RESET}",         flush=True )
  
        global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts )

        if DEBUG>0:
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )

      elif args.cases == 'ALL_ELIGIBLE_CASES':

        target                = 'image_test'
        cases_required        = n_samples
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}P_C_GENERATE:   INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) n_tiles (this run)................................................................. = {MIKADO}{n_tiles}{RESET}",                flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (just_test) cases_required .................................................................... = {MIKADO}{cases_required}{RESET}",         flush=True )
  
        global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts )

        if DEBUG>0:
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )



    else:

      #  (2B)   Generate Training dataset

      if args.cases=='UNIMODE_CASE':
        
        # (2Ba) case_designation_flag for training set = UNIMODE_CASE____IMAGE
        #       case_designation_flag for test     set = UNIMODE_CASE____IMAGE_TEST
      
        test_cases      = int( n_samples * pct_test )
        training_cases  = n_samples - test_cases
                      
        for target in [ 'image_train', 'image_test' ]:
    
          if target=='image_train':
            cases_required        =  training_cases
            case_designation_flag =  'UNIMODE_CASE____IMAGE'
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}P_C_GENERATE:   INFO:    about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) n_samples (this run)............................................................... = {MIKADO}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) n_tiles   (this run)............................................................... = {MIKADO}{n_tiles}{RESET}",                                  flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) cases_required (training cases = int(n_samples * (1 - pct_test ) ) ................ = {MIKADO}{cases_required}{RESET}",                           flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) hence tiles required for training = cases_required * n_tiles ) .................... = {MIKADO}{cases_required * n_tiles}{RESET}",                 flush=True )

          if target=='image_test':
            cases_required        =  test_cases
            case_designation_flag =  'UNIMODE_CASE____IMAGE_TEST'
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}P_C_GENERATE:   INFO:    about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_test) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_test) n_samples (this run)............................................................... = {MIKADO}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_test) n_tiles   (this run)............................................................... = {MIKADO}{n_tiles}{RESET}",                                  flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_test) pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_test) cases_required (test cases = n_samples - training_cases) .......................... = {MIKADO}{cases_required}{RESET}",                           flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}P_C_GENERATE:   INFO: (image_train) hence tiles required for in-training testing = test cases * n_tiles ) ............ = {MIKADO}{cases_required * n_tiles}{RESET}",                 flush=True )
    
          global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts )

        if DEBUG>0:
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )


        # (2Ba) case_designation_flag for training set = args.cases
        #       case_designation_flag for test set     = args.cases
        #       both training and test sets will be drawn from the same set of examples
      
      elif args.cases == 'ALL_ELIGIBLE_CASES':

        target                = 'image_train'
        cases_required        =  n_samples
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{WHITE}P_C_GENERATE:   INFO:    about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}{CLEAR_LINE}",  flush=True )
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    cases_required (this run).......................................................... = {MIKADO}{n_samples}{RESET}{CLEAR_LINE}",              flush=True )
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}{CLEAR_LINE}",               flush=True )
  
        global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts )

        if DEBUG>0:
          print ( f"{DULL_WHITE}P_C_GENERATE:   INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )

    return ( SUCCESS )
      
      
      
      
      

  
  
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvOLD VERSION - DON'T REMOVE YET (NOW USED FOR PRE-TRAINING (-p option )

  # (2) process IMAGE data if applicable
  
  
  if ( input_mode=='image' ) & ( pretrain=='True' ):  
  
  #  (2A) set up numpy data structures to accumulate image data
  
    tiles_required = n_samples*n_tiles
  
    images_new      = np.ones ( ( tiles_required,  3, tile_size, tile_size ), dtype=np.uint8   )              
    fnames_new      = np.zeros( ( tiles_required                           ), dtype=np.int64   )              # np.int64 is equiv of torch.long
    img_labels_new  = np.zeros( ( tiles_required,                          ), dtype=np.int_    )              # img_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 

  
  #  (2B) Process image data 

    # check to see that there actually are tiles to process    
    if cumulative_png_file_count==0:
      print ( f"{RED}P_C_GENERATE:    FATAL:  there are no tile files ('png' files) at all. To generate tiles, run '{CYAN}./do_all.sh -d <cancer type code> -i image{RESET}{RED}' ... halting now{RESET}", flush=True )                 
      sys.exit(0)         

    if ( input_mode=='image' ):
      print( f"\r{ORANGE}P_C_GENERATE:   INFO:    input_mode is '{RESET}{CYAN}{input_mode}{RESET}{ORANGE}', so rna and other data will not be generated{RESET}" )  
    
    # process image data
    tiles_processed         =  0
    directories_processed   =  0

    #(2C) Traverse dataset and process pytorch data file
    
    designated_case_count   = 0
    
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration of os.walk takes us to a new directory under data_dir    

      if DEBUG>66:
        print( f"{PALE_GREEN}P_C_GENERATE:   INFO:   now processing case (directory) {CYAN}{dir_path}{RESET}" )

      use_this_case_flag=False
      try:
        fqn = f"{dir_path}/{args.cases}"        
        f = open( fqn, 'r' )
        use_this_case_flag=True
        if DEBUG>6:
          print ( f"{PALE_GREEN}P_C_GENERATE:   INFO:   case                            {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[100C is a {BITTER_SWEET}{args.cases}{RESET}{PALE_GREEN} case  \r\033[160C (count= {designated_case_count+1}{RESET}{PALE_GREEN})",  flush=True )
        designated_case_count+=1
      except Exception:
        pass

      # Does the work
      if ( use_this_case_flag==True ) | ( args.cases=='ALL_ELIGIBLE_CASES' ):
          tiles_processed = process_image_files ( args, dir_path, dirs, files, images_new, img_labels_new, fnames_new, n_tiles, tiles_processed )

      directories_processed+=1
      
      if tiles_processed>=tiles_required:
        break

    if DEBUG>0:
      print( f"P_C_GENERATE:   INFO:     directories_processed          = {BLEU}{directories_processed-1:<8d}{RESET}",  flush=True       )
      print( f"P_C_GENERATE:   INFO:     tiles_processed                = {BLEU}{tiles_processed:<8d}{RESET}",          flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     tiles required (notional)      = {BLEU}{tiles_required:<8d}{RESET}",           flush=True       )    

    images_new      = images_new     [0:tiles_processed]   # trim
    img_labels_new  = img_labels_new [0:tiles_processed]   # trim
    fnames_new      = fnames_new     [0:tiles_processed]   # trim

    if DEBUG>2:
      print( f"P_C_GENERATE:   INFO:     images_new.shape               = {GOLD}{images_new.shape}{RESET}",             flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     img_labels_new.shape           = {GOLD}{img_labels_new.shape}{RESET}",         flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     fnames_new.shape               = {GOLD}{fnames_new.shape}{RESET}",             flush=True       )

    if DEBUG>0:
      print( "P_C_GENERATE:   INFO:    now converting data and labels from numpy array to Torch tensor")  

    # convert everything into Torch style tensors
  
    images_new   = torch.Tensor( images_new )
    fnames_new   = torch.Tensor( fnames_new ).long()
    fnames_new.requires_grad_( False )
    img_labels_new  = torch.Tensor( img_labels_new ).long()                                                # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
    img_labels_new.requires_grad_( False )                                                                 # labels aren't allowed gradients

    if DEBUG>0:
      print( "P_C_GENERATE:   INFO:    finished converting data and labels from numpy array to Torch tensor") 

  
    if DEBUG>6:
        print ( f"P_C_GENERATE:   INFO:    img_labels_new                =                             {MIKADO}{img_labels_new.numpy()}{RESET}{CLEAR_LINE}"    )  
  
    # save as torch '.pth' file for subsequent loading by dataset function
  
    
 
    fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_image_train.pth"                            # save the same dataset for both train and test ( we split just 'train' in loader, and we won't be using test)

    torch.save({
        'images':     images_new,
        'fnames':     fnames_new,
        'img_labels': img_labels_new,
    }, fqn )

    if DEBUG>0:  
      print( f"P_C_GENERATE:   INFO:    Torch dictionary saved to {MAGENTA}{fqn}{RESET}")


    fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_image_test.pth"
    
    torch.save({
        'images':     images_new,
        'fnames':     fnames_new,
        'img_labels': img_labels_new,
    }, fqn )
    
    
    if DEBUG>0:  
      print( f"P_C_GENERATE:   INFO:    Torch dictionary saved to {MAGENTA}{fqn}{RESET}")

    return ( SUCCESS )    

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^OLD VERSION - DON'T REMOVE YET (NOW USED FOR PRE-TRAINING (-p option )





  # (3) process "IMAGE_RNA" data, if applicable, with generation of concatenated embeddings as a preliminary step 

  if ( input_mode=='image_rna' ):

    print( f"{CARRIBEAN_GREEN}P_C_GENERATE:   INFO:  input_mode is '{CYAN}{input_mode}{RESET}{CARRIBEAN_GREEN}', so image and other data will not be generated{RESET}" )  
    
      
    # (3A) preliminary step: create concatenated image+rna embeddings

    dirs_which_have_matched_image_rna_files      = 0
    dirs_which_dont_have_matched_image_rna_files = 0
    designated_case_count                        = 0    
    
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>888:  
        print( f"{DIM_WHITE}P_C_GENERATE:   INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
      has_matched_image_rna_data = False
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
                
        try:
          fqn = f"{dir_path}/_image_rna_matched___rna.npy"                                                 # if it has an rna embedding file, it must have both image and rna data
          f = open( fqn, 'r' )
          has_matched_image_rna_data=True
          if DEBUG>6:
            print ( f"{PALE_GREEN}P_C_GENERATE:   INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[103C has both matched and rna files (listed above)  \r\033[200C (count= {dirs_which_have_matched_image_rna_files+1}{RESET}{PALE_GREEN})",  flush=True )
          dirs_which_have_matched_image_rna_files+=1
        except Exception:
          if DEBUG>6:
            print ( f"{PALE_RED}P_C_GENERATE:   INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_RED} \r\033[103C DOES NOT have both matched and rna files (listed above)  \r\033[200C (count= {dirs_which_dont_have_matched_image_rna_files+1}{RESET}{PALE_RED})",  flush=True )
            dirs_which_dont_have_matched_image_rna_files+=1

        designated_case_flag = False
        if has_matched_image_rna_data==True:
          try:
            fqn = f"{dir_path}/{args.cases}"
            if DEBUG>6:
              print ( f"{PURPLE}P_C_GENERATE:   INFO:   fqn = {CYAN}{fqn}{RESET}",  flush=True )
            f = open( fqn, 'r' )
            designated_case_flag=True
            if DEBUG>6:
              print ( f"{PURPLE}P_C_GENERATE:   INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PURPLE} \r\033[103C is a designated case \r\033[150C ({CYAN}case flag = {MAGENTA}{args.cases}{RESET}{PURPLE} \r\033[200C (count= {designated_case_count+1}{RESET}{PURPLE})",  flush=True )
            designated_case_count+=1
          except Exception:
            if DEBUG>6:
              print ( f"{PALE_RED}P_C_GENERATE:   INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_RED} \r\033[103C is NOT a designated case \r\033[150C ({CYAN}case = {MAGENTA}{args.cases}{RESET}{PALE_RED})",  flush=True )


                  
        if has_matched_image_rna_data & designated_case_flag:
          
          for f in sorted( files ):
            
            if f.endswith( args.embedding_file_suffix_image ):
              if DEBUG>6:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:   image embedding file        =  {ARYLIDE}{f}{RESET}",  flush=True )
              image_embedding = np.load ( f"{dir_path}/{f}" )
              rna_embedding   = np.load ( f"{dir_path}/_image_rna_matched___rna.npy" )
              #rna_embedding = np.zeros_like( rna_embedding )                                              # force rna-seq portion of the concatenated embedding to zero for testing purposes
              if DEBUG>6:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:   image_embedding.shape       =  {ARYLIDE}{image_embedding.shape}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:   rna_embedding.shape         =  {BLEU}{rna_embedding.shape}{RESET}",  flush=True )
              
              image_rna_embedding = np.concatenate( (image_embedding, rna_embedding ), axis=0)
              if DEBUG>6:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:   image_rna_embedding.shape   =  {BITTER_SWEET}{image_rna_embedding.shape}{RESET}",  flush=True )
              random_name   = f"_{random.randint(10000000, 99999999)}_image_rna_matched___image_rna"
              save_fqn      = f"{dir_path}/{random_name}"
              if DEBUG>6:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:   saving concatenated embedding {BITTER_SWEET}{save_fqn}{RESET}",  flush=True )
              np.save ( save_fqn, image_rna_embedding )
              

    # (3B) determine 'n_genes' (sic) by looking at an (any) image_rna file, (so that it doesn't have to be manually entered as a user parameter)
    if use_autoencoder_output=='False':
   
      if DEBUG>0:
        print ( f"P_C_GENERATE:   INFO:  about to determine length of an image_rna embedding"      )
    
      found_one=False
      for dir_path, dirs, files in os.walk( data_dir ):                                                    # each iteration takes us to a new directory under data_dir
        if not (dir_path==data_dir):                                                                       # the top level directory is  skipped because it only contains sub-directories, not data
          
          if check_mapping_file( args, dir_path ) == True:
          
            for f in sorted(files):                                                                        # examine every file in the current directory
              if found_one==True:
                break
              if  ( f.endswith( args.embedding_file_suffix_image_rna  ) ):
                if DEBUG>999:
                  print (f)     
                image_rna_file      = os.path.join(dir_path, f )
                try:
                  image_rna = np.load( image_rna_file )
                  n_genes=image_rna.shape[0]
                  found_one=True
                  if DEBUG>0:
                    print ( f"P_C_GENERATE:   INFO:  image_rna.shape      =  '{MIKADO}{image_rna.shape}{RESET}' "      )
                  if DEBUG>0:
                    print ( f"P_C_GENERATE:   INFO:  n_genes (determined)  = {MIKADO}{n_genes}{RESET}"        )
                except Exception as e:
                    print ( f"{RED}GENERATE: FATAL: '{e}'{RESET}" )
                    print ( f"{PALE_RED}GENERATE: FATAL: Explanation: a requiPALE_RED image_rna embedding file doesn't exist. (Probably no image_rna files exist){RESET}" )                 
                    print ( f"{PALE_RED}GENERATE: FATAL:              did you change to image_rna mode from another input mode but neglect to run '{CYAN}./do_all.sh{RESET}{PALE_RED}' to generate the image_rna files the network needs for image_rna mode ? {RESET}" )
                    print ( f"{PALE_RED}GENERATE: FATAL:              if so, run '{CYAN}./do_all.sh -d <cancer type code> -i image_rna{RESET}{PALE_RED}' to generate the image_rna files{RESET}" )                 
                    print ( f"{PALE_RED}GENERATE: FATAL:              halting now ...{RESET}" )                 
                    sys.exit(0)

      if found_one==False:                  
        print ( f"{RED}P_C_GENERATE:       FATAL: No image_rna embedding files exist in the dataset directory ({MAGENTA}{data_dir}{RESET}{RED})"                                                                          )                 
        print ( f"{PALE_RED}P_C_GENERATE:              Possible explanations:{RESET}"                                                                                                                       )
        print ( f"{PALE_RED}P_C_GENERATE:                (1) Did you change to {CYAN}image_rna{RESET}{PALE_RED} mode from another input mode but neglect to regenerate the files input requiPALE_RED for {CYAN}image_rna{RESET}{PALE_RED} mode ?" )
        print ( f"{PALE_RED}P_C_GENERATE:              Possible remedies:{RESET}"                                                                                                                       )
        print ( f"{PALE_RED}P_C_GENERATE:                    (A) (easist, but regenerates everything) run '{CYAN}./do_all_image_rna.sh.sh  -d <cancer_type_code>{RESET}{PALE_RED}" )
        print ( f"{PALE_RED}P_C_GENERATE:                    (B) (faster) run '{CYAN}./do_all.sh     -d <cancer_type_code> -i rna{RESET}{PALE_RED}' to train the rna model'" )                 
        print ( f"{PALE_RED}P_C_GENERATE:                            then run '{CYAN}./just_test.sh  -d <cancer_type_code> -i rna  -m image_rna <cancer_type_code> image_rna{RESET}{PALE_RED}' to generate the rna embedding files'" )                 
        print ( f"{PALE_RED}P_C_GENERATE:                            then run '{CYAN}./do_all.sh     -d <cancer_type_code> -i image_rna{RESET}{PALE_RED}' to generate the image_rna embedding files. " )                 
        print ( f"{PALE_RED}P_C_GENERATE:            Halting now{RESET}" )                 
        sys.exit(0)
    
                


    # (3C) set up numpy data structures to accumulate image_rna data as it is processed 

    # need to know required_number_of_image_rna_files in advance to be able to create numpy array to hold them. Determine using this rule: one concatenated image_rna file (___image_rna.npy) will be created for every existing IMAGE embedding file in a directory that has both an image embedding file (___image.npy)and an rna embedding file (___rna.npy)         
    if DEBUG>0:
      print ( f"{ORANGE}P_C_GENERATE:   INFO:   dirs_which_have_matched_image_rna_files         =  {BITTER_SWEET}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )
      print ( f"{ORANGE}P_C_GENERATE:   INFO:   n_tiles                                         =  {BITTER_SWEET}{n_tiles}{RESET}",                                  flush=True )
    required_number_of_image_rna_files = dirs_which_have_matched_image_rna_files * n_tiles
    if DEBUG>0:
      print ( f"{ORANGE}P_C_GENERATE:   INFO:   (hence) required_number_of_image_rna_files      =  {BITTER_SWEET}{required_number_of_image_rna_files}{RESET}",  flush=True )


    if ( input_mode=='image_rna' ):

      global_image_rna_files_processed =  0                                                                                         # global count of genes processed
      
      if use_autoencoder_output=='False':
        genes_new      = np.zeros( ( required_number_of_image_rna_files, 1, n_genes                ), dtype=np.float64 )
      fnames_new       = np.zeros( ( required_number_of_image_rna_files                            ), dtype=np.int64   )              
      gnames_new       = np.zeros( ( required_number_of_image_rna_files                            ), dtype=np.uint8   )            # was gene names                                               NOT USED
      rna_labels_new   = np.zeros( ( required_number_of_image_rna_files,                           ), dtype=np.int_    )            # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training



    # (3D) process image_rna data
    
    # process image_rna data
    
    symlinks_created               = 0
            
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under data_dir
 
      if DEBUG>9:  
        print( f"{DIM_WHITE}P_C_GENERATE:   INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
        
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
        
        # ~ designated_multimode_case_flag_found=False
        # ~ try:
          # ~ fqn = f"{dir_path}/MULTIMODE____TEST"        
          # ~ f = open( fqn, 'r' )
          # ~ designated_multimode_case_flag_found=True
          # ~ if DEBUG>6:
            # ~ print ( f"{PALE_GREEN}P_C_GENERATE:   INFO:   case                            {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[100C is a {BITTER_SWEET}designated multimode{RESET}{PALE_GREEN} case  \r\033[160C (count= {designated_multimode_case_flag_count+1}{RESET}{PALE_GREEN})",  flush=True )
          # ~ designated_multimode_case_flag_count+=1
        # ~ except Exception:
          # ~ pass
  
                  
        # ~ if has_matched_image_rna_data:        
  
          # ~ # (3Di) Make and store a  softlink based on an integer reference to the case id for later use so that DENSE will later know where to save the rna-seq embeddings (if this option is enabled)
          
          # ~ for f in sorted( files ):  
              
            # ~ if  f.endswith( args.embedding_file_suffix_image_rna ):
              
              # ~ image_rna_file_link_id   = random.randint(1000000, 9999999)                                    # generate random string to use for the softlink to the file name (can't have strings in tensors)
              # ~ image_rna_file_link_name = f"{image_rna_file_link_id:d}"
        
              # ~ fqcd = f"{dir_path}"                                                                           # fully qualified case directory
              # ~ fqln = f"{args.data_dir}/{image_rna_file_link_name}.fqln"                                      # fully qualified link name
              # ~ try:
                # ~ os.symlink( fqcd, fqln)                                                                      # make a link from fqln to fqcd
                # ~ symlinks_created +=1
                # ~ if DEBUG>88:
                  # ~ print ( f"P_C_GENERATE:       ALL GOOD:          softlink (fqln) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )
              # ~ except Exception as e:
                # ~ print ( f"{RED}P_C_GENERATE:    EXCEPTION RAISED:  softlink (fqln) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )
                
              # ~ if DEBUG>0:
                # ~ print( f"P_C_GENERATE:   INFO:                      image_rna_file_link_id =  {MAGENTA}{image_rna_file_link_id}{RESET}" )
                # ~ print( f"P_C_GENERATE:   INFO:                    image_rna_file_link_name = '{MAGENTA}{image_rna_file_link_name}{RESET}'" )
                # ~ print (f"P_C_GENERATE:   INFO: fully qualified file name of case directory = '{MAGENTA}{fqcd}{RESET}'" )        
                # ~ print (f"P_C_GENERATE:   INFO:            symlink for referencing the fqcd = '{MAGENTA}{fqln}{RESET}'" )
                # ~ print (f"P_C_GENERATE:   INFO:                     symlinks created so far = '{MAGENTA}{symlinks_created}{RESET}'" )
  
  
          # (3Dii) Process the rna-seq file
          
          for f in sorted( files ):              
  
            if  f.endswith( args.embedding_file_suffix_image_rna  ):
              
              image_rna_file         = os.path.join( dir_path, f         )
              label_file             = os.path.join( dir_path, class_numpy_file_name )
  
              # set up the pytorch array
              if DEBUG>8:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:  file                         = {BLEU}{f}{RESET}", flush=True )

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS
              
              # ~ if use_autoencoder_output=='False':                                                            # skip gene processing (but do labels and gnames) if we're using autoencoder output 


# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS

            
              try:
                image_rna_embedding = np.load( image_rna_file )
                if DEBUG>99:
                  print ( f"P_C_GENERATE:   INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                  print ( f"P_C_GENERATE:   INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"P_C_GENERATE:   INFO:         image_rna_embedding             =  '{image_rna_embedding}' "            )
                  print ( f"P_C_GENERATE:   INFO:         genes_new       =  '{genes_new}' "      )
              except Exception as e:
                print ( f"{RED}P_C_GENERATE:    FATAL: {e} ... halting now [118]{RESET}" )
                sys.exit(0)
          
          
              if DEBUG>999:  
                print( f"P_C_GENERATE:   INFO:                     image_rna_embedding = {CYAN}{image_rna_embedding}{RESET}" )              

          
              genes_new [global_image_rna_files_processed] =  np.transpose(image_rna_embedding)
                
              if DEBUG>99:
                print ( f"P_C_GENERATE:   INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                print ( f"P_C_GENERATE:   INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"P_C_GENERATE:   INFO:         image_rna_embedding             =  \n'{MIKADO}{np.transpose(image_rna_embedding[1,:])}{RESET}' "      )
                print ( f"P_C_GENERATE:   INFO:         genes_new [{global_image_rna_files_processed}] =  '{CYAN}{genes_new[global_image_rna_files_processed]}{RESET}' ")                       
                
              try:
                label = np.load( label_file)
                if DEBUG>6:
                  print ( "P_C_GENERATE:   INFO:         label       =  \"{:}\"".format(  label      ) )
                if DEBUG>0:
                  print ( f"{label[0]},", end=', ', flush=True )
              except Exception as e:
                print ( f"{RED}P_C_GENERATE:           FATAL: '{e}'{RESET}" )
                print ( f"{PALE_RED}P_C_GENERATE:           FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{PALE_RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{PALE_RED}{RESET}" )
                print ( f"{PALE_RED}P_C_GENERATE:           FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{PALE_RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{PALE_RED}' to regenerate them{RESET}" ) 
                print ( f"{PALE_RED}P_C_GENERATE:           FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{PALE_RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
                print ( f"{PALE_RED}P_C_GENERATE:           FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{PALE_RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{PALE_RED}', because without it, no class files can be generated'{RESET}" )                                    
                print ( f"{PALE_RED}P_C_GENERATE:           FATAL:  cannot continue - halting now{RESET}" )                 
                sys.exit(0)     
                
              rna_labels_new[global_image_rna_files_processed] =  label[0]
  
              
              if DEBUG>6:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:         label[0][{MIKADO}{global_image_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
            
            
             # fnames_new [global_image_rna_files_processed  ]  =  image_rna_file_link_id                  # link to folder from which that this image_rna_embedding sample belongs to - passed in as a parameter
              fnames_new [global_image_rna_files_processed  ]  =  777   
            
              if DEBUG>888:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:        image_rna_file_link_id = {MIKADO}{image_rna_file_link_id}{RESET}",                          flush=True )
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:        fnames_new[{MIKADO}{global_image_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_image_rna_files_processed  ]}{RESET}", flush=True )
              
              
              gnames_new [global_image_rna_files_processed]  =  781                                        # Any old number. We don't currently use these
            
              if DEBUG>888:
                print ( f"{WHITE}P_C_GENERATE:   INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
                time.sleep(.4)  
  
              global_image_rna_files_processed+=1
  
              if DEBUG>9:
                print ( f"{WHITE}P_C_GENERATE:   INFO: global_image_rna_files_processed = {MIKADO}{global_image_rna_files_processed}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO: n_samples                  = {CYAN}{n_samples}{RESET}",               flush=True )
  
          
          
          
      
          
  if ( input_mode=='rna' ):


    # (4A) determine 'n_genes' by looking at an (any) rna file, (so that it doesn't have to be manually entered as a user parameter)

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS
    
    # ~ if use_autoencoder_output=='False':
  
# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS

    # To determine n_genes, (so that it doesn't have to be manually specified), need to examine just ONE of the rna files   
    if DEBUG>2:
      print ( f"P_C_GENERATE:   INFO:  about to determine value of 'n_genes'"      )
  
    found_one=False
    for dir_path, dirs, files in os.walk( data_dir ):                                                    # each iteration takes us to a new directory under data_dir
      if not (dir_path==data_dir):                                                                       # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
        
        if check_mapping_file( args, dir_path ) == True:
        
          for f in sorted(files):                                                                        # examine every file in the current directory
            if found_one==True:
              break
            if ( f.endswith( tcga_rna_seq_file_suffix[1:]) ):                                                     # have to leave out the asterisk apparently
              if DEBUG>999:
                print (f)     
              rna_file      = os.path.join(dir_path, rna_file_name)
              try:
                rna = np.load( rna_file )
                n_genes=rna.shape[0]
                found_one=True
                if DEBUG>9:
                  print ( f"P_C_GENERATE:   INFO:   rna.shape             =  '{MIKADO}{rna.shape}{RESET}' "      )
                if DEBUG>2:
                  print ( f"P_C_GENERATE:   INFO:  n_genes (determined)  = {MIKADO}{n_genes}{RESET}"        )
              except Exception as e:
                  print ( f"{RED}P_C_GENERATE:    FATAL:   error message: '{e}'{RESET}" )
                  print ( f"{PALE_RED}P_C_GENERATE:    FATAL:   explanation: a required rna file doesn't exist. (Probably no rna files exist){RESET}" )                 
                  print ( f"{PALE_RED}P_C_GENERATE:    FATAL:   did you change from image mode to rna mode but neglect to run '{CYAN}./do_all.sh{RESET}{PALE_RED}' to generate the rna files the NN needs for rna mode ? {RESET}" )
                  print ( f"{PALE_RED}P_C_GENERATE:    FATAL:   if so, run '{CYAN}./do_all.sh -d <cancer type code> -i rna{RESET}{PALE_RED}' to generate the rna files{RESET}" )                 
                  print ( f"{PALE_RED}P_C_GENERATE:    FATAL:   halting now ...{RESET}" )                 
                  sys.exit(0)

    if found_one==False:                  
      print ( f"{RED}P_C_GENERATE:    FATAL: No rna files at all exist in the dataset directory ({MAGENTA}{data_dir}{RESET}{RED})"                                                                          )                 
      print ( f"{PALE_RED}P_C_GENERATE:              Possible explanations:{RESET}"                                                                                                                       )
      print ( f"{PALE_RED}P_C_GENERATE:                (1) The dataset '{CYAN}{args.dataset}{RESET}{PALE_RED}' doesn't have any rna-seq data. It might only have image data{RESET}" )
      print ( f"{PALE_RED}P_C_GENERATE:                (2) Did you change from image mode to rna mode but neglect to run '{CYAN}./do_all.sh{RESET}{PALE_RED}' to generate the files requiPALE_RED for rna mode ? {RESET}" )
      print ( f"{PALE_RED}P_C_GENERATE:                    If so, run '{CYAN}./do_all.sh <cancer_type_code> rna{RESET}{PALE_RED}' to generate the rna files{RESET}{PALE_RED}. After that, you will be able to use '{CYAN}./just_run.sh <cancer_type_code> rna{RESET}{PALE_RED}'" )                 
      print ( f"{PALE_RED}P_C_GENERATE:            Halting now{RESET}" )                 
      sys.exit(0)


    # (4B) set up numpy data structures to accumulate rna data as it is processed 

    if ( input_mode=='rna' ):
      # set up numpy data structures to accumulate rna-seq data as it is processed
      
# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS          
      # ~ if use_autoencoder_output=='False':
        # ~ genes_new      = np.zeros( ( n_samples, 1, n_genes                ), dtype=np.float64 )

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS     
     
      genes_new      = np.zeros( ( n_samples, 1, n_genes                ), dtype=np.float64 )
        
      fnames_new       = np.zeros( ( n_samples                            ), dtype=np.int64   )              
      gnames_new       = np.zeros( ( n_samples                            ), dtype=np.uint8   )              # was gene names                                               NOT USED
      global_rna_files_processed =  0                                                                        # global count of rna files processed
      rna_labels_new   = np.zeros( ( n_samples,                           ), dtype=np.int_    )              # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training


    # (4C) process rna-seq data
    
    # info and warnings
    if ( input_mode=='rna' ):
      print( f"P_C_GENERATE:   INFO:  input_mode is '{RESET}{CYAN}{input_mode}{RESET}', so image and other data will not be generated{RESET}" )  

    if use_unfiltered_data==True:
      rna_suffix = tcga_rna_seq_file_suffix[1:]
      print( f"{ORANGE}P_C_GENERATE:   INFO:  flag {CYAN}'USE_UNFILTERED_DATA'{CYAN}{RESET}{ORANGE} is set, so all genes listed in file '{CYAN}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}{ORANGE}' will be used{RESET}" )        
    else:
      rna_suffix = rna_file_reduced_suffix
      print( f"{ORANGE}P_C_GENERATE:   INFO:  The subset of genes specified in '{CYAN}TARGET_GENES_REFERENCE_FILE{RESET}{ORANGE}' = '{CYAN}{args.target_genes_reference_file}{RESET}{ORANGE}' will be used.{RESET}" ) 
      print( f"{ORANGE}P_C_GENERATE:   INFO:  Set user parameter {CYAN}'USE_UNFILTERED_DATA'{RESET}{ORANGE} to {MIKADO}True{RESET}{ORANGE} if you wish to use all genes (specifically, all the genes in '{MAGENTA}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}')" ) 
   

    not_designated_case_count  = 0
    designated_case_count      = 0
    
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under data_dir

      if DEBUG>888:  
        print( f"{DIM_WHITE}P_C_GENERATE:   INFO:   now processing case (directory) {CYAN}{dir_path}{RESET}" )
          
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data

        use_this_case_flag=False
        try:
          fqn = f"{dir_path}/{args.cases}"        
          f = open( fqn, 'r' )
          designated_case_count+=1
          use_this_case_flag=True
          if DEBUG>10:
            print ( f"{PALE_GREEN}P_C_GENERATE:   INFO:   case                            {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[132C is a {BITTER_SWEET}designated {RESET}{PALE_GREEN} case (designated_case_count = {designated_case_count}{RESET}",  flush=True )
        except Exception:
          not_designated_case_count+=1
          if DEBUG>10:
            print ( f"{DARK_RED}P_C_GENERATE:   INFO:   case                            {RESET}{CYAN}{dir_path}{RESET}{DARK_RED} \r\033[132C is NOT    a {BITTER_SWEET}designated {RESET}{DARK_RED} case (not_designated_case_count = {not_designated_case_count}{RESET})",  flush=True )


        if ( use_this_case_flag==True ) | ( args.cases=='ALL_ELIGIBLE_CASES' ):   # ALL_ELIGIBLE_CASES is not positively defined

          for f in sorted( files ):
                                     
            if DEBUG>999:
              print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:  rna_suffix                   = {MIKADO}{rna_suffix}{RESET}", flush=True )
              print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:  file                         = {MAGENTA}{f}{RESET}",         flush=True )            
                                    
            # (4Ci) Make and store a  softlink based on an integer reference to the case id for later use so that DENSE will later know where to save the rna-seq embeddings (if this option is enabled)
              
            if  f == rna_file_name:                                                                          # always 'rna.npy'

              fqcd                      = f"{dir_path}"
              parent_dir                = os.path.split(fqcd)[1]
              no_special_chars_version  = re.sub('[^A-Za-z0-9]+', '', parent_dir).lstrip()
              final_chars               = no_special_chars_version[-6:]
              int_version               = int( final_chars, 16)
                  
              if DEBUG>10:
                print (f"P_C_GENERATE:   INFO:              fully qualified case directory = '{MAGENTA}{fqcd}{RESET}'" )
                print (f"P_C_GENERATE:   INFO:                                    dir_path = '{MAGENTA}{dir_path}{RESET}'" )
                print (f"P_C_GENERATE:   INFO:                                  parent_dir = '{MAGENTA}{parent_dir}{RESET}'" )
                print (f"P_C_GENERATE:   INFO:                    no_special_chars_version = '{MAGENTA}{no_special_chars_version}{RESET}'" )
                print (f"P_C_GENERATE:   INFO:                                 final_chars = '{MAGENTA}{final_chars}{RESET}'" )
                print (f"P_C_GENERATE:   INFO:                                 hex_version = '{MAGENTA}{int_version}{RESET}'" )
        
              rna_file_link_id   = int_version
              rna_file_link_name = f"{rna_file_link_id:d}"
        
              fqln = f"{args.data_dir}/{rna_file_link_name}.fqln"                                                  # name for the link
              try:
                os.symlink( fqcd, fqln)                                                                            # make the link
              except Exception as e:
                if DEBUG>2:
                  print ( f"{ORANGE}P_C_GENERATE:   INFO:  Link already exists{RESET}" )
                else:
                  pass
        
              if DEBUG>10:
                print( f"P_C_GENERATE:   INFO:                            rna_file_link_id =  {MAGENTA}{rna_file_link_id}{RESET}" )
                print( f"P_C_GENERATE:   INFO:                          rna_file_link_name = '{MAGENTA}{rna_file_link_name}{RESET}'" )
                print( f"P_C_GENERATE:   INFO:                                        fqln = '{MAGENTA}{fqln}{RESET}'" )


              
              # ~ rna_file_link_id   = random.randint(1000000, 9999999)                                          # generate random string to use for the softlink to the file name (can't have strings in tensors)
              # ~ rna_file_link_name = f"{rna_file_link_id:d}"
        
              # ~ fqcd = f"{dir_path}"                                                                           # fully qualified case directory
              # ~ fqln = f"{args.data_dir}/{rna_file_link_name}.fqln"                                            # fully qualified link name
              try:
                os.symlink( fqcd, fqln)                                                                      # make a link from fqln to fqcd
                if DEBUG>55:
                  print ( f"P_C_GENERATE:   INFO:       softlink (fqln) {MAGENTA}{fqln}{RESET} \r\033[93C and target (fqcd) = {MAGENTA}{fqcd}{RESET}" )
              except Exception as e:
                if DEBUG>55:
                  print ( f"{ORANGE}P_C_GENERATE:   INFO:       softlink (fqln) {MAGENTA}{fqln}{RESET}{ORANGE} \r\033[93C to target (fqcd) = {MAGENTA}{fqcd}{RESET}{ORANGE} \r\033[185C already exists, which may well be fine and intended{RESET}" )
  

            # (4Cii) Process the rna-seq file
              
              rna_file      = os.path.join( dir_path, rna_file_name         )
              label_file    = os.path.join( dir_path, class_numpy_file_name )
  
              # set up the pytorch array
            
              if DEBUG>10:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:  file                         = {BLEU}{f}{RESET}", flush=True )

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS     
              
              # ~ if use_autoencoder_output=='False':                                                          # Skip gene processing (but do labels and gnames) if we're using autoencoder output. We'll LATER load and use ae output file as genes_new rather than process raw rna-seq data 

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS     

          
              try:
                rna = np.load( rna_file )
                if DEBUG>99:
                  print ( f"P_C_GENERATE:   INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                  print ( f"P_C_GENERATE:   INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"P_C_GENERATE:   INFO:         rna             =  '{rna}' "            )
                  print ( f"P_C_GENERATE:   INFO:         genes_new       =  '{genes_new}' "      )
              except Exception as e:
                print ( f"{RED}P_C_GENERATE:    FATAL: {e} ... halting now [2118]{RESET}" )
                sys.exit(0)
          
          
              if DEBUG>999:  
                print( f"P_C_GENERATE:   INFO:                     rna = {CYAN}{rna}{RESET}" )              
              
              rna[np.abs(rna) < 1] = 0                                                                   # set all the values lower than 1 to be 0
              
              if gene_data_transform=='NONE':
                transformed_rna = rna                                  
              elif gene_data_transform=='LN':
                transformed_rna = np.log(rna) 
              elif gene_data_transform=='LOG2':
                transformed_rna = np.log2(rna) 
              elif gene_data_transform=='LOG2PLUS1':
                transformed_rna = np.log2(rna+1)
              elif gene_data_transform=='LOG10':
                transformed_rna = np.log10(rna)
              elif gene_data_transform=='LOG10PLUS1':
                transformed_rna = np.log10(rna+1)
              else:
                print( f"{RED}P_C_GENERATE:   FATAL:        no such gene data transformation as: {gene_data_transform} ... halting now[184]{RESET}" )
                sys.exit(0) 
          
              if gene_data_norm=='NONE':
                normalized_rna =  transformed_rna
              elif gene_data_norm=='JUST_SCALE':
                normalized_rna = transformed_rna / np.std(transformed_rna)   
              elif gene_data_norm=='GAUSSIAN':
                normalized_rna = ( transformed_rna - np.mean(transformed_rna) ) / np.std(transformed_rna)                                             
              else:
                print( f"{RED}P_C_GENERATE:   FATAL:        no such gene normalization mode as: {gene_data_norm} ... halting now[378]{RESET}" )  
                sys.exit(0)       
          
              genes_new [global_rna_files_processed] =  np.transpose(normalized_rna)               
                
              if DEBUG>99:
                print ( f"P_C_GENERATE:   INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                print ( f"P_C_GENERATE:   INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"P_C_GENERATE:   INFO:         rna             =  \n'{MIKADO}{np.transpose(rna[1,:])}{RESET}' "      )
                print ( f"P_C_GENERATE:   INFO:         genes_new [{global_rna_files_processed}] =  '{CYAN}{genes_new[global_rna_files_processed]}{RESET}' ")                       
              
              try:
                label = np.load( label_file)
                if DEBUG>99:
                  print ( "P_C_GENERATE:   INFO:         label.shape =  \"{:}\"".format(  label.shape) )
                  print ( "P_C_GENERATE:   INFO:         label       =  \"{:}\"".format(  label      ) )
                if DEBUG>10:
                  print ( f"{label[0]},", end='', flush=True )
                if label[0]>highest_class_number:
                  if DEBUG>1:
                    print ( f"{ORANGE}P_C_GENERATE:   INFO: label is greater than '{CYAN}HIGHEST_CLASS_NUMBER{RESET}' - - skipping this example (label = {MIKADO}{label[0]}{ORANGE}){RESET}"      )
                  break
              except Exception as e:
                print ( f"{RED}P_C_GENERATE:         FATAL: '{e}'{RESET}" )
                print ( f"{RED}P_C_GENERATE:         FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{RED}{RESET}" )
                print ( f"{RED}P_C_GENERATE:         FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{RED}' to regenerate them{RESET}" ) 
                print ( f"{RED}P_C_GENERATE:         FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
                print ( f"{RED}P_C_GENERATE:         FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{RED}', because without it, no class files can be generated'{RESET}" )                                    
                print ( f"{RED}P_C_GENERATE:         FATAL:  cannot continue - halting now{RESET}" )                 
                sys.exit(0)     

  
                
              rna_labels_new[global_rna_files_processed] =  label[0]
              #rna_labels_new[global_rna_files_processed] =  random.randint(0,5)                        ################### swap truth labels to random numbers for testing purposes
              
              if DEBUG>777:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:        rna_labels_new[{MIKADO}{global_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
            
            
              fnames_new [global_rna_files_processed  ]  =  rna_file_link_id                               # link to folder from which that this rna sample belongs to - passed in as a parameter
            
              if DEBUG>888:
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:        rna_file_link_id = {MIKADO}{rna_file_link_id}{RESET}",                          flush=True )
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO:        fnames_new[{MIKADO}{global_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_rna_files_processed  ]}{RESET}", flush=True )
              
              
              gnames_new [global_rna_files_processed]  =  443                                              # Any old number. We don't currently use these
            
              if DEBUG>888:
                print ( f"{WHITE}P_C_GENERATE:   INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
                time.sleep(.4)  
  
              global_rna_files_processed+=1
  
              if DEBUG>55:
                print ( f"{WHITE}P_C_GENERATE:   INFO: global_rna_files_processed = {MIKADO}{global_rna_files_processed}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}P_C_GENERATE:   INFO: n_samples                  = {CYAN}{n_samples}{RESET}",               flush=True )
   
      if  global_rna_files_processed>=n_samples:
        break


    if ( args.cases=='ALL_ELIGIBLE_CASES' ):
      case_count =  global_rna_files_processed
    else:
      case_count =  designated_case_count

    genes_new       = genes_new      [0:case_count]
    rna_labels_new  = rna_labels_new [0:case_count]
    fnames_new      = fnames_new     [0:case_count]


    if DEBUG>0:
      print( f"P_C_GENERATE:   INFO:     case_count                     = {GOLD}{case_count}{RESET}",                   flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     genes_new.shape                = {GOLD}{genes_new.shape}{RESET}",              flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     rna_labels_new.shape           = {GOLD}{rna_labels_new.shape}{RESET}",         flush=True       ) 
      print( f"P_C_GENERATE:   INFO:     fnames_new.shape               = {GOLD}{fnames_new.shape}{RESET}",             flush=True       )

    if args.n_samples[0] > case_count:
      print( f"{ORANGE}P_C_GENERATE:   WARN: proposed number of samples {CYAN}N_SAMPLES{RESET}{ORANGE} (= {MIKADO}{args.n_samples[0]}{ORANGE}) is greater than the number of cases processed, 'case_count' ( = {MIKADO}{case_count}{RESET}{ORANGE}){RESET}" )
      print( f"{ORANGE}P_C_GENERATE:   WARN: now changing {CYAN}args.n_samples[0]){ORANGE} to {MIKADO}{case_count}{RESET}{RESET}" )
      print( f"{ORANGE}P_C_GENERATE:   WARN: explanation: perhaps you specified a flag such as {CYAN}MULTIMODE____TEST{RESET}{ORANGE}, which selects a subset of the available samples, and this subset is smaller that {CYAN}{n_samples}{RESET}{ORANGE}. This is perfectly fine.{RESET}" )
      args.n_samples[0] = case_count

    if args.batch_size[0] > case_count:
      print( f"{ORANGE}P_C_GENERATE:   WARN: proposed batch size ({CYAN}BATCH_SIZE{RESET} = {MIKADO}{args.batch_size[0]}{RESET}{ORANGE}) is greater than the number of cases available, 'case_count'  ( = {MIKADO}{case_count}{RESET}{ORANGE})" )
      print( f"{ORANGE}P_C_GENERATE:   WARN: changing {CYAN}args.batch_size[0]){CYAN} to {MIKADO}{int(0.2*global_rna_files_processed)}{RESET}" )
      print( f"{ORANGE}P_C_GENERATE:   WARN: further comment: If you don't like this value of {CYAN}BATCH_SIZE{RESET}{ORANGE}, stop the program and provide a new value in the configuration file {MAGENTA}conf.py{RESET}")
      args.batch_size[0] = int(0.2*global_rna_files_processed)






#############

  if ( input_mode=='rna' )  | ( input_mode=='image_rna'):
 
 
 # THIS IS DIFFERENT TO THE CLASSIFY VERSION - WE ARE SAVING A COPY OF THE ENTIRE GENES DATASET FOR USE IN filter_genes() AND IN analyse_data()
 
    # convert to pandas dataframe, then pickle and save for possible use with analyse_data
     
    ensg_reference_file_name = f"{data_dir}/ENSG_reference"
    if DEBUG>0:  
      print ( f"P_C_GENERATE:   INFO:      ensg_reference_file_name (containing genes ENSG names to be used as column headings) = {MAGENTA}{ensg_reference_file_name}{RESET}", flush=True )
      print ( f"P_C_GENERATE:   INFO:      about to add pandas column headings for the genes dataframe  {RESET}" )       
    with open( ensg_reference_file_name ) as f:
      ensg_reference = f.read().splitlines()
    df = pd.DataFrame(np.squeeze( genes_new ), columns=ensg_reference)
    
    if DEBUG>9:
      print ( f"P_C_GENERATE:   INFO:       len(ensg_reference.shape) = {MAGENTA}{len(ensg_reference)}{RESET}", flush=True ) 
      print ( f"P_C_GENERATE:   INFO:       df.shape = {MAGENTA}{df.shape}{RESET}", flush=True )   
    if DEBUG>9:
      print (df)
    
    # save a pickled pandas version
    save_file_name  = f'{args.base_dir}/{args.application_dir}/modes/analyse_data/genes.pickle'
    print( f"P_C_GENERATE:   INFO:      about to label, squeeze, convert to pandas dataframe, pickle and save {MIKADO}'genes_new'{RESET} to {MAGENTA}{save_file_name}{RESET}" )   
    df.to_pickle( save_file_name )  
    print( f"P_C_GENERATE:   INFO:      finished labeling, converting to dataframe, pickling and saving       {MIKADO}'genes_new'{RESET} to {MAGENTA}{save_file_name}{RESET}" )
    
    # save a pickled cupy version. we'll lose the headers because numpy and cupy are number-only data structures
    save_file_name  = f'{args.base_dir}/{args.application_dir}/modes/analyse_data/genes_cupy.pickle.npy'
    if DEBUG>0:
      print ( f"P_C_GENERATE:   INFO:      converting pandas dataframe to numpy array", flush=True ) 
    df_npy = df.to_numpy()                                                                                # convert pandas dataframe to numpy
    if DEBUG>0:
      print ( f"P_C_GENERATE:   INFO:      converting numpy array to cupy array", flush=True )     
    df_cpy = cupy.asarray( df_npy )
    if DEBUG>0:
      print ( f"P_C_GENERATE:   INFO:      saving cupy array to {MAGENTA}{save_file_name}{RESET} for possible subsequent use in {CYAN}analyse_data{RESET} mode", flush=True )
    cupy.save( save_file_name, df_cpy, allow_pickle=True) 
   
 # THIS IS DIFFERENT TO THE DLBCL VERSION - WE ARE SAVING A COPY OF THE ENTIRE GENES DATASET FOR USE IN filter_genes() AND IN analyse_data()
 
 
 
 
 
# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS     
    
    # ~ if use_autoencoder_output=='True':                                                                     # then we already have them in Torch format, in the ae feature file, which we now load

      # ~ fpath = '%s/ae_output_features.pt' % args.log_dir
      # ~ if DEBUG>0:
        # ~ print( f"{BRIGHT_GREEN}P_C_GENERATE:   INFO:  about to load autoencoder generated feature file from {MAGENTA}{fpath}{RESET}", flush=True )
      # ~ try:
        # ~ genes_new    = torch.load( fpath )
        # ~ genes_new    = genes_new.unsqueeze(1)                                                              # add a dimension to make it compatible with existing (non-autoencoder) code
        # ~ n_genes      = genes_new.shape[2]                                                                  # i.e. number of gene-like-features from the dimensionality reduced output of the autoencoder
        # ~ args.n_genes = n_genes
        # ~ if DEBUG>0:
          # ~ print ( f"P_C_GENERATE:   INFO:    genes_new.size         = {MIKADO}{genes_new.size()}{RESET}"      ) 
          # ~ print ( f"P_C_GENERATE:   INFO:    n_genes  (determined)  = {MIKADO}{n_genes}{RESET}"               )               
# ~ #        genes_new = np.ones( ( n_samples, 1, n_genes                 ), dtype=np.float64 )
# ~ #        if DEBUG>0:
# ~ #          print ( f"P_C_GENERATE:   INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")        
        # ~ if DEBUG>0:   
          # ~ print( f"{BRIGHT_GREEN}P_C_GENERATE:   INFO:  autoencoder feature file successfully loaded{RESET}" )          
      # ~ except Exception as e:
        # ~ print ( f"{RED}P_C_GENERATE:   INFO:  could now load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
        # ~ if DEBUG>0:
          # ~ print ( f"{RED}P_C_GENERATE:   INFO:  the exception was: {CYAN}'{e}'{RESET}" )
        # ~ sys.exit(0)

    # ~ else:          
      # ~ genes_new    = torch.Tensor( genes_new   )
# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE ALWAYS WANT TO GENERATE A PYTORCH DICTIONARY IN THE CASE OF PRECOMPRESS     


    genes_new    = torch.Tensor( genes_new   )
    
    gnames_new   = torch.Tensor( gnames_new  ) 
    gnames_new.requires_grad_( False )        
    rna_labels_new  = torch.Tensor(rna_labels_new).long()                                                  # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
    rna_labels_new.requires_grad_( False )                                                                 # labels aren't allowed gradients

    if  ( args.just_test!='True' ):
      if len(np.unique(rna_labels_new)) != len(class_names):
        print ( f"{RED}P_C_GENERATE:    FATAL:    there are fewer cancer types in the cases to be trained than there are in configuration parameter {CYAN}CLASS_NAMES{RESET}{RESET}"  ) 
        print ( f"{RED}P_C_GENERATE:    FATAL:      classes represented     = {MIKADO}{len(np.unique(rna_labels_new))}{RESET}"                                   ) 
        print ( f"{RED}P_C_GENERATE:    FATAL:      classes in {CYAN}CLASS_NAMES{RESET}{RED}  = {MIKADO}{len(class_names)}{RESET}"                                 ) 
        print ( f"{RED}P_C_GENERATE:    FATAL:      possible remedy (1) include more cases so that it will be more likely that examples of the missing class(es) will be represented{RESET}"  )
        print ( f"{RED}P_C_GENERATE:    FATAL:      possible remedy (2) edit {CYAN}CLASS_NAMES{RESET}{RED} to only include the names of classes actually represented (but be careful: the order of {CYAN}CLASS_NAMES{RESET}{RED} has to be the same as the order of the class labels as represented in the master spreadsheet {CYAN}{args.cancer_type}_mapping_file_MASTER{RESET}{RED}, and 'gaps' are not permitted" )
        print ( f"{RED}P_C_GENERATE:    FATAL:    halting now ...{RESET}", flush=True)
        sys.exit(0)    

    if DEBUG>8:
      print ( f"P_C_GENERATE:   INFO:  finished converting rna   data and labels     from numpy array to Torch tensor")
      print ( f"P_C_GENERATE:   INFO:    Torch size of genes_new       =  (~samples)                   {MIKADO}{genes_new.size()}{RESET}"      )
      print ( f"P_C_GENERATE:   INFO:    Torch size of gnames_new      =  (~samples)                   {MIKADO}{gnames_new.size()}{RESET}"     )
      print ( f"P_C_GENERATE:   INFO:    Torch size of rna_labels_new  =  (~samples)                   {MIKADO}{rna_labels_new.size()}{RESET}" )
  
    if DEBUG>88:
        print ( f"P_C_GENERATE:   INFO:    fnames_new                    =                               {MIKADO}{fnames_new}{RESET}"             )
  
    if DEBUG>0:        
        print ( f"P_C_GENERATE:   INFO:    rna_labels_new                =                             {MIKADO}{rna_labels_new.numpy()}{RESET}"    )     



# THIS IS DIFFERENT TO THE DLBCL VERSION

    fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_rna_train.pth"

    torch.save({
        'genes':      genes_new,
        'fnames':     fnames_new,
        'gnames':     gnames_new, 
        'rna_labels': rna_labels_new,           
    }, fqn )

    if DEBUG>0:  
      print( f"P_C_GENERATE:   INFO:    Torch dictionary saved to {MAGENTA}{fqn}{RESET}")


    fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_rna_test.pth"                               # same dataset for both training and test modes in the case of pre_compress. We want to use the optimised model to compress every sample

    torch.save({
        'genes':      genes_new,
        'fnames':     fnames_new,
        'gnames':     gnames_new, 
        'rna_labels': rna_labels_new,           
    }, fqn )

    if DEBUG>0:  
      print( f"P_C_GENERATE:   INFO:    Torch dictionary saved to {MAGENTA}{fqn}{RESET}")

# THIS IS DIFFERENT TO THE DLBCL VERSION


  
    # (5) RETURN
  
    return ( n_genes )





#----------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
#----------------------------------------------------------------------------------------------------------

def check_mapping_file ( args, case ):

  if args.mapping_file_name=='none':
    return True
  else:                                                                                                    # a custom mapping file is active
  # then look inside the custom mapping file to see if this case exists
    exists = True
    return ( exists )


#----------------------------------------------------------------------------------------------------------
def process_image_files ( args, dir_path, dirs, files, images_new, img_labels_new, fnames_new, n_tiles, global_tiles_processed ):

  # 1  find the SVS file in each directory then  make and store an integer reference to it (which will include the case id) so for later use when we are displaying tiles that belong to it

  for f in sorted (files):                                                                                 # examine every file in the current directory

    if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
      
      fqsn                      = f"{dir_path}/entire_patch.npy"
      parent_dir                = os.path.split(os.path.dirname(fqsn))[1]
      no_special_chars_version  = re.sub('[^A-Za-z0-9]+', '', parent_dir).lstrip()
      final_chars               = no_special_chars_version[-6:]
      int_version               = int( final_chars, 16)
          
      if DEBUG>5:
        print (f"P_C_GENERATE:   INFO:  fully qualified file name of slide = '{MAGENTA}{fqsn}{RESET}'" )
        print (f"P_C_GENERATE:   INFO:                            dir_path = '{MAGENTA}{dir_path}{RESET}'" )
        print (f"P_C_GENERATE:   INFO:                          parent_dir = '{MAGENTA}{parent_dir}{RESET}'" )
        print (f"P_C_GENERATE:   INFO:            no_special_chars_version = '{MAGENTA}{no_special_chars_version}{RESET}'" )
        print (f"P_C_GENERATE:   INFO:                         final_chars = '{MAGENTA}{final_chars}{RESET}'" )
        print (f"P_C_GENERATE:   INFO:                         hex_version = '{MAGENTA}{int_version}{RESET}'" )


      svs_file_link_id   = int_version
      svs_file_link_name = f"{svs_file_link_id:d}"

      fqln = f"{args.data_dir}/{svs_file_link_name}.fqln"                                                  # name for the link
      try:
        os.symlink( fqsn, fqln)                                                                            # make the link
      except Exception as e:
        if DEBUG>2:
          print ( f"{ORANGE}P_C_GENERATE:   INFO:  Link already exists{RESET}" )
        else:
          pass

      if DEBUG>5:
        print( f"P_C_GENERATE:   INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}" )
        print( f"P_C_GENERATE:   INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'" )
        print( f"P_C_GENERATE:   INFO:                                fqln = '{MAGENTA}{fqln}{RESET}'" )



  # 2  set up the pytorch array (using the parameters that were passed in: images_new, img_labels_new, fnames_new

  
  tile_extension  = "png"
  tiles_processed = 0

  for f in sorted( files ):                                                                                # examine every file in the current directory
           
    if DEBUG>999:
      print( f"P_C_GENERATE:   INFO:               files                  = {MAGENTA}{files}{RESET}"      )
  
    image_file    = os.path.join(dir_path, f)
    label_file    = os.path.join(dir_path, args.class_numpy_file_name)


    if DEBUG>66:  
      print( f"P_C_GENERATE:   INFO:     image_file    = {CYAN}{image_file}{RESET}", flush=True   )
      print( f"P_C_GENERATE:   INFO:     label_file    = {CYAN}{label_file}{RESET}",   flush=True   )

    
    if ( f.endswith('.' + tile_extension ) & (not ( 'mask' in f ) ) & (not ( 'ized' in f ) )   ):          # because there may be other png files in each image folder besides the tile image files

      try:
        img = cv2.imread( image_file )
        if DEBUG>66:
          print ( f"P_C_GENERATE:   INFO:     image_file    =  {BLEU}{image_file}{RESET}"    )
      except Exception as e:
        print ( f"{RED}P_C_GENERATE:          FATAL: when processing: '{image_file}'{RESET}", flush=True)    
        print ( f"{RED}P_C_GENERATE:                 reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}P_C_GENERATE:                 halting now{RESET}", flush=True)
        sys.exit(0)    


      try:
        images_new [global_tiles_processed,:] =  np.moveaxis(img, -1,0)                                    # add it to the images array
      except Exception as e:
        print ( f"{RED}P_C_GENERATE:          FATAL:  [33124] reported error was: '{e}'{RESET}", flush=True )
        print ( f"{RED}P_C_GENERATE:                   Explanation: The dimensions of the array reserved for tiles is  {MIKADO}{images_new [global_tiles_processed].shape}{RESET}{RED}; whereas the tile dimensions are: {MIKADO}{np.moveaxis(img, -1,0).shape}{RESET}", flush=True )                 
        print ( f"{RED}P_C_GENERATE:                   {RED}Did you change the tile size without regenerating the tiles? {RESET}", flush=True )
        print ( f"{RED}P_C_GENERATE:                   {RED}Either run'{CYAN}./do_all.sh -d <cancer type code> -i image{RESET}{RED}' to generate {MIKADO}{images_new [global_tiles_processed].shape[1]}x{images_new [global_tiles_processed].shape[1]}{RESET}{RED} tiles, or else change '{CYAN}TILE_SIZE{RESET}{RED}' to {MIKADO}{np.moveaxis(img, -1,0).shape[1]}{RESET}", flush=True )                 
        print ( f"{RED}P_C_GENERATE:                   {RED}Halting now{RESET}", flush=True )                 
        sys.exit(0)

      try:                                                                                                 # every tile has an associated label - the same label for every tile image in the directory
        label = np.load( label_file )
        if DEBUG>99:
          print ( f"P_C_GENERATE:   INFO:     label.shape   =  {MIKADO}{label.shape}{RESET}"   )
        if DEBUG>66:         
          print ( f"P_C_GENERATE:   INFO:     label value   =  {MIKADO}{label[0]}{RESET}"      )
        if DEBUG>2:
          print ( f"{label[0]},", end='', flush=True )
      except Exception as e:
        print ( f"{RED}P_C_GENERATE:          FATAL: when processing: '{label_file}'{RESET}", flush=True)        
        print ( f"{RED}P_C_GENERATE:                 reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}P_C_GENERATE:                 halting now{RESET}", flush=True)
        sys.exit(0)
        
      img_labels_new[global_tiles_processed] =  label[0]                                                   # add it to the labels array
      #img_labels_new[global_tiles_processed] =  random.randint(0,5)                                       # swap truth labels to random numbers for testing purposes

      if DEBUG>77:  
        print( f"P_C_GENERATE:   INFO:     label                  = {MIKADO}{label[0]:<8d}{RESET}", flush=True   )
        

      fnames_new [global_tiles_processed]  =  svs_file_link_id                                             # link to filename of the slide from which this tile was extracted - see above

      if DEBUG>99:
          print( f"P_C_GENERATE:   INFO: symlink for tile (fnames_new [{BLEU}{global_tiles_processed:3d}{RESET}]) = {BLEU}{fnames_new [global_tiles_processed]}{RESET}" )
      

      if DEBUG>66:
        print ( "=" *180)
        print ( "P_C_GENERATE:   INFO:          tile {:} for this image:".format( global_tiles_processed+1))
        print ( "P_C_GENERATE:   INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
        print ( "P_C_GENERATE:   INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
      if DEBUG>99:
        print ( "P_C_GENERATE:   INFO:                value = \n{:}".format(images_new[global_tiles_processed]))

      the_class=img_labels_new[global_tiles_processed]
      if the_class>3000:
          print ( f"{RED}P_C_GENERATE:    FATAL: Ludicrously large class value detected (class={MIKADO}{the_class}{RESET}{RED}) for tile '{MAGENTA}{image_file}{RESET}" )
          print ( f"{RED}P_C_GENERATE:    FATAL: hanting now [1718]{RESET}" )
          sys.exit(0)
          
      if DEBUG>9:
        size_in_bytes=img_labels_new[global_tiles_processed].size * img_labels_new[global_tiles_processed].itemsize
        print ( f"P_C_GENERATE:   INFO:        for img_labels_new[{global_tiles_processed}]; class={the_class}" )

      if DEBUG>66:
        print ( "P_C_GENERATE:   INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
        print ( "P_C_GENERATE:   INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
        print ( "P_C_GENERATE:   INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
       
      global_tiles_processed+=1
      
      tiles_processed+=1
      if tiles_processed==n_tiles:
        break
      
    else:
      if DEBUG>1:
        print( f"P_C_GENERATE:   INFO:          other file = {MIKADO}{image_file}{RESET}".format(  ) )
        
    
  if DEBUG>66:
    print( f"P_C_GENERATE:   INFO:                              tiles processed in in directory: '{MAGENTA}{dir_path}{RESET}' = {ARYLIDE}{tiles_processed:<8d}{RESET}",        flush=True       )   
    
  if  ( args.just_test!='True' ):
    if (tiles_processed!=n_tiles) & (tiles_processed!=0):
      print( f"{RED}P_C_GENERATE:   INFO:     tiles processed in directory: '{MAGENTA}{dir_path}{RESET}' = \r\033[150C{MIKADO}{tiles_processed:<8d}{RESET}{RED}\r\033[180C<<<<<<<<<<<< anomoly {RESET}", flush=True  )       
      time.sleep(4)
  
  return global_tiles_processed
  

  
#----------------------------------------------------------------------------------------------------------
def generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tile_size, class_counts  ):

  
  #  These are all the valid cases:
  #       
  #  user flag:
  # -c ALL_ELIGIBLE_CASES                      <<< Largest possible set. For use in unimode experiments only (doesn't set aside test cases for multimode):      for STAD: total image examples - UNIMODE_CASE____IMAGE_TEST
  # -c UNIMODE_CASE               <<< Largest set that can be used in multimode experiments (because it  uses ummatched cases for unimode runs):   for STAD: total image examples - UNIMODE_CASE____IMAGE_TEST - MULTIMODE____TEST
  # -c UNIMODE_CASE____IMAGE      <<< Same as UNIMODE_CASE. Convenience only, but permitted.
  # -c UNIMODE_CASE____MATCHED            <<< Combination to use when testing the thesis (uses only matched cases for unimode runs):                       for STAD: total matched examples - DESIGNATED_UNIMODE_CASE____IMAGE_TEST - MULTIMODE____TEST
  # -c MULTIMODE____TEST          <<< Use for MULTIMODE testing. These cases are guaranteed to have never been seen during UNIMODE testing
  #
  #  What to generate as the training set:
  #  If  -c = ...
  #    ALL_ELIGIBLE_CASES                !UNIMODE_CASE____IMAGE_TEST                                           <<< NOT currently catered for temp. ALL_ELIGIBLE_CASES____IMAGE_TEST_FLAG not currently segmented, so use UNIMODE_CASE____IMAGE_TEST temporarily
  #    UNIMODE_CASE         UNIMODE_CASE____IMAGE          &! MULTIMODE____TEST      <<< currently catered for
  #    UNIMODE_CASE____MATCHED      DESIGNATED_UNIMODE_CASE____IMAGE       &! MULTIMODE____TEST      <<< NOT currently catered for. Neither flag currently exists. 
  #    MULTIMODE____TEST                  N/A                                                                  <<< Never used for training
  #
  #  What to generate as the test set:
  #  If -c = ...
  #    ALL_ELIGIBLE_CASES                UNIMODE_CASE____IMAGE_TEST                                             <<< NOT currently catered for.      
  #    UNIMODE_CASE         UNIMODE_CASE____IMAGE_TEST                                             <<< currently catered for
  #    UNIMODE_CASE____MATCHED      DESIGNATED_UNIMODE_CASE____IMAGE_TEST                                          <<< should be catered for, but I'm not sure it's working. need to test firhter
  #    MULTIMODE____TEST    MULTIMODE____TEST
  #
  #  Tiling implications:
  #
  #  ------------------------------------------+-----------------------------------------------------------------------------------------------------+----------------------------------------------------
  #                 User Flag                  |                                             Training                                                |                      Test                         
  #  ------------------------------------------+-----------------------------------------------------------------------------------------------------+----------------------------------------------------
  #                                count:      |               1 - (pct_test * n_samples)       |               (pct_test * n_samples)               |         cases_reserved_for_image_rna
  #  ------------------------------------------+------------------------------------------------+----------------------------------------------------+----------------------------------------------------
  #                                            |                                                |                                                    |
  #  -c ALL_ELIGIBLE_CASES                     |          !MULTIMODE____TEST       |      UNIMODE_CASE____IMAGE_TEST       |         MULTIMODE____TEST
  #                                            |                                                |                                                    |
  #  -c UNIMODE_CASE              |       UNIMODE_CASE____IMAGE       |      UNIMODE_CASE____IMAGE_TEST       |         MULTIMODE____TEST
  #                                            |                                                |                                                    |
  #  -c UNIMODE_CASE____MATCHED           |     DESIGNATED_UNIMODE_CASE____IMAGE      |    DESIGNATED_UNIMODE_CASE____IMAGE_TEST      |         MULTIMODE____TEST
  #                                            |                                                |                                                    |
  #  ------------------------------------------+------------------------------------------------+----------------------------------------------------+----------------------------------------------------
  #  -c MULTIMODE____TEST         |                                                |                                                    |         MULTIMODE____TEST
  #  -------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------

  tiles_required  = cases_required*n_tiles
  
  images_new      = np.ones ( ( tiles_required,  3, tile_size, tile_size ), dtype=np.uint8   )              
  fnames_new      = np.zeros( ( tiles_required                           ), dtype=np.int64   )              # np.int64 is equiv of torch.long
  img_labels_new  = np.zeros( ( tiles_required,                          ), dtype=np.int_    )              # img_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 

  if DEBUG>0:
    print( f"P_C_GENERATE:   INFO:     images_new.shape               = {PINK}{images_new.shape}{RESET}",             flush=True       ) 
    print( f"P_C_GENERATE:   INFO:     img_labels_new.shape           = {PINK}{img_labels_new.shape}{RESET}",         flush=True       ) 
    print( f"P_C_GENERATE:   INFO:     fnames_new.shape               = {PINK}{fnames_new.shape}{RESET}",             flush=True       )


  global_tiles_processed  = 0
  designated_case_count   = 0
  directories_processed   = 0
  sufficient_cases_found  = False
  
  for dir_path, dirs, files in os.walk( args.data_dir ):    


    tiles_processed = 0                                                                                    # count of tiles processed for just this case

    #  (1) is it a one of the cases we're looking for ?    

    use_this_case_flag=False
    if not (dir_path==args.data_dir):                                                                      # the top level directory is skipped because it only contains sub-directories, not data      
      
      use_this_case_flag=False
      try:
        fqn = f"{dir_path}/SLIDE_TILED"        
        f = open( fqn, 'r' )
        if DEBUG>4:
          print ( f"{PALE_GREEN}P_C_GENERATE:   INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[130C has been tiled{RESET}{CLEAR_LINE}",  flush=True )
        if case_designation_flag=='ALL_ELIGIBLE_CASES':
          use_this_case_flag=True
        try:
          fqn = f"{dir_path}/{case_designation_flag}"        
          f = open( fqn, 'r' )
          use_this_case_flag=True
          if DEBUG>8:
            print ( f"\n{GREEN}P_C_GENERATE:   INFO:   case \r\033[55C'{COTTON_CANDY}{dir_path}{RESET}{GREEN}' \r\033[130C is     a case flagged as '{CYAN}{case_designation_flag}{RESET}{GREEN}' - - including{RESET}{CLEAR_LINE}",  flush=True )
        except Exception:
          if DEBUG>4:
            print ( f"{RED}P_C_GENERATE:   INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED} \r\033[130C is not a case flagged as '{CYAN}{case_designation_flag}{RESET}{RED}' - - skipping{RESET}{CLEAR_LINE}",  flush=True )
      except Exception:
        if DEBUG>4:
          print ( f"{PALE_RED}P_C_GENERATE:   INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C has not been tiled{RESET}{CLEAR_LINE}",  flush=True )

      try:                                                                                                 # every tile has an associated label - the same label for every tile image in the directory
        label_file    = os.path.join(dir_path, args.class_numpy_file_name)
        label = np.load( label_file )
        if label[0]>highest_class_number:
          use_this_case_flag=False
          if DEBUG>0:
            print ( f"{ORANGE}P_C_GENERATE:   INFO: label is greater than '{CYAN}HIGHEST_CLASS_NUMBER{RESET}' - - skipping this example (label = {MIKADO}{label[0]}{ORANGE}){RESET}"      )
          pass
      except Exception as e:
        print ( f"{RED}P_C_GENERATE:          FATAL: when processing: '{label_file}'{RESET}", flush=True)        
        print ( f"{RED}P_C_GENERATE:                 reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}P_C_GENERATE:                 halting now{RESET}", flush=True)
        sys.exit(0)
    
    
    if ( use_this_case_flag==True ):

      if DEBUG>3:
        print( f"{CARRIBEAN_GREEN}P_C_GENERATE:   INFO:   now processing case:           '{CAMEL}{dir_path}{RESET}'" )
        
        
      # (2) Set up symlink

      for f in sorted( files ):  

        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
          
          fqsn                      = f"{dir_path}/entire_patch.npy"
          parent_dir                = os.path.split(os.path.dirname(fqsn))[1]
          no_special_chars_version  = re.sub('[^A-Za-z0-9]+', '', parent_dir).lstrip()
          final_chars               = no_special_chars_version[-6:]
          int_version               = int( final_chars, 16)
              
          if DEBUG>5:
            print (f"P_C_GENERATE:   INFO:  fully qualified file name of slide = '{MAGENTA}{fqsn}{RESET}'"                     )
            print (f"P_C_GENERATE:   INFO:                            dir_path = '{MAGENTA}{dir_path}{RESET}'"                 )
            print (f"P_C_GENERATE:   INFO:                          parent_dir = '{MAGENTA}{parent_dir}{RESET}'"               )
            print (f"P_C_GENERATE:   INFO:            no_special_chars_version = '{MAGENTA}{no_special_chars_version}{RESET}'" )
            print (f"P_C_GENERATE:   INFO:                         final_chars = '{MAGENTA}{final_chars}{RESET}'"              )
            print (f"P_C_GENERATE:   INFO:                         hex_version = '{MAGENTA}{int_version}{RESET}'"              )
    
    
          svs_file_link_id   = int_version
          svs_file_link_name = f"{svs_file_link_id:d}"
    
          fqln = f"{args.data_dir}/{svs_file_link_name}.fqln"                                                  # name for the link
          try:
            os.symlink( fqsn, fqln)                                                                            # make the link
          except Exception as e:
            if DEBUG>2:
              print ( f"{ORANGE}P_C_GENERATE:   INFO:  Link already exists{RESET}" )
            else:
              pass
    
          if DEBUG>8:
            print( f"P_C_GENERATE:   INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}"          )
            print( f"P_C_GENERATE:   INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'"       )
            print( f"P_C_GENERATE:   INFO:                                fqln = '{MAGENTA}{fqln}{RESET}'"                     )   


      # (3) set up the array for each png entry in this directory
      
      tile_extension  = "png"
    
      for f in sorted( files ):                                                                                # examine every file in the current directory
               
        if DEBUG>999:
          print( f"P_C_GENERATE:   INFO:               files                  = {MAGENTA}{files}{RESET}"      )
      
        image_file    = os.path.join(dir_path, f)
    
        if DEBUG>2:  
          print( f"P_C_GENERATE:   INFO:     image_file    = {CYAN}{image_file}{RESET}", flush=True   )
          print( f"P_C_GENERATE:   INFO:     label_file    = {CYAN}{label_file}{RESET}",   flush=True   )
    
        
        if ( f.endswith('.' + tile_extension ) & (not ( 'mask' in f ) ) & (not ( 'ized' in f ) )   ):          # because there may be other png files in each image folder besides the tile image files
    
          try:
            img = cv2.imread( image_file )
            if DEBUG>2:
              print ( f"P_C_GENERATE:   INFO:     image_file    =  \r\033[55C'{AMETHYST}{image_file}{RESET}'"    )
          except Exception as e:
            print ( f"{RED}P_C_GENERATE:          FATAL: when processing: '{image_file}'{RESET}", flush=True)    
            print ( f"{RED}P_C_GENERATE:                 reported error was: '{e}'{RESET}", flush=True)
            print ( f"{RED}P_C_GENERATE:                 halting now{RESET}", flush=True)
            sys.exit(0)    
    
    
          try:
            images_new [global_tiles_processed,:] =  np.moveaxis(img, -1,0)                                    # add it to the images array
          except Exception as e:
            print ( f"{RED}P_C_GENERATE:          FATAL:  [1320] reported error was: '{e}'{RESET}", flush=True )
            print ( f"{RED}P_C_GENERATE:                   Explanation: The dimensions of the array reserved for tiles is  {MIKADO}{images_new [global_tiles_processed].shape}{RESET}{RED}; whereas the tile dimensions are: {MIKADO}{np.moveaxis(img, -1,0).shape}{RESET}", flush=True )                 
            print ( f"{RED}P_C_GENERATE:                   {RED}Did you change the tile size without regenerating the tiles? {RESET}", flush=True )
            print ( f"{RED}P_C_GENERATE:                   {RED}Either run'{CYAN}./do_all.sh -d <cancer type code> -i image{RESET}{RED}' to generate {MIKADO}{images_new [global_tiles_processed].shape[1]}x{images_new [global_tiles_processed].shape[1]}{RESET}{RED} tiles, or else change '{CYAN}TILE_SIZE{RESET}{RED}' to {MIKADO}{np.moveaxis(img, -1,0).shape[1]}{RESET}", flush=True )                 
            print ( f"{RED}P_C_GENERATE:                   {RED}Halting now{RESET}", flush=True )                 
            sys.exit(0)
    
          try:                                                                                                 # every tile has an associated label - the same label for every tile image in the directory
            label = np.load( label_file )
            if DEBUG>99:
              print ( f"P_C_GENERATE:   INFO:     label.shape   =  {MIKADO}{label.shape}{RESET}"   )
            if DEBUG>99:         
              print ( f"P_C_GENERATE:   INFO:     label value   =  {MIKADO}{label[0]}{RESET}"      )
            if DEBUG>2:
              print ( f"{AMETHYST}{label[0]}", end=', ', flush=True )
          except Exception as e:
            print ( f"{RED}P_C_GENERATE:          FATAL: when processing: '{label_file}'{RESET}", flush=True)        
            print ( f"{RED}P_C_GENERATE:                 reported error was: '{e}'{RESET}", flush=True)
            print ( f"{RED}P_C_GENERATE:                 halting now{RESET}", flush=True)
            sys.exit(0)
                                    
          img_labels_new[global_tiles_processed] =  label[0]                                                   # add it to the labels array
          class_counts[label[0]]+=1                                                                            # keep track of the number of examples of each class 
          
          #img_labels_new[global_tiles_processed] =  random.randint(0,5)                                       # swap truth labels to random numbers for testing purposes
    
          if DEBUG>77:  
            print( f"P_C_GENERATE:   INFO:     label                  = {MIKADO}{label[0]:<8d}{RESET}", flush=True   )
            
    
          fnames_new [global_tiles_processed]  =  svs_file_link_id                                             # link to filename of the slide from which this tile was extracted - see above
    
          if DEBUG>88:
              print( f"P_C_GENERATE:   INFO: symlink for tile (fnames_new [{BLEU}{global_tiles_processed:3d}{RESET}]) = {BLEU}{fnames_new [global_tiles_processed]}{RESET}" )
          
    
          if DEBUG>66:
            print ( "=" *180)
            print ( "P_C_GENERATE:   INFO:          tile {:} for this image:".format( global_tiles_processed+1))
            print ( "P_C_GENERATE:   INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
            print ( "P_C_GENERATE:   INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
          if DEBUG>99:
            print ( "P_C_GENERATE:   INFO:                value = \n{:}".format(images_new[global_tiles_processed]))
    
          the_class=img_labels_new[global_tiles_processed]
          if the_class>3000:
              print ( f"{RED}P_C_GENERATE:    FATAL: Ludicrously large class value detected (class={MIKADO}{the_class}{RESET}{RED}) for tile '{MAGENTA}{image_file}{RESET}" )
              print ( f"{RED}P_C_GENERATE:    FATAL: hanting now [1718]{RESET}" )
              sys.exit(0)
              
    
          if DEBUG>66:
            print ( "P_C_GENERATE:   INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
            print ( "P_C_GENERATE:   INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
            print ( "P_C_GENERATE:   INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
           
          global_tiles_processed+=1
          
          tiles_processed+=1
          if tiles_processed==n_tiles:
            break
          
        else:
          if DEBUG>44:
            print( f"P_C_GENERATE:   INFO:          other file = {MIKADO}{image_file}{RESET}".format(  ) )

        
      if DEBUG>2:
        print( f"P_C_GENERATE:   INFO:   tiles processed in directory: \r\033[55C'{MAGENTA}{dir_path}{RESET}'  \r\033[130C= {MIKADO}{tiles_processed:<8d}{RESET}",        flush=True       )   
        
      if args.just_test=='False':
        if tiles_processed!=n_tiles:
          print( f"{RED}P_C_GENERATE:   INFO:   tiles processed in directory:  \r\033[55C'{MAGENTA}{dir_path}{RESET}' = {MIKADO}{tiles_processed:<8d}{RESET}{RED}\r\033[180C<<<<<<<<<<<< anomoly {RESET}", flush=True  )       
          time.sleep(2)

      directories_processed+=1
      if DEBUG>8:
        print( f"{CAMEL}P_C_GENERATE:   INFO:   directories_processed = {CAMEL}{directories_processed:<4d}{RESET}",  flush=True        )   
      if directories_processed>=cases_required:
        sufficient_cases_found=True
        if DEBUG>0:
          print( f"{CAMEL}P_C_GENERATE:   INFO:   sufficient directories were found ({CYAN}cases_required{RESET}{CAMEL} = {MIKADO}{cases_required:<4d}{RESET})",  flush=True        )  
        break

  if sufficient_cases_found!=True:
    if args.just_test=='True':
      print( f"{RED}P_C_GENERATE:    ERROR:   (test mode) the number of cases found and processed ({MIKADO}{directories_processed}{RESET}{RED}) is less than the number requested ({MIKADO}{cases_required}{RESET}{RED})",  flush=True        ) 
      print( f"{RED}P_C_GENERATE:    ERROR:   (test mode)   possible explanation: if you set '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{RED}' to a number less than the number of classes actually present in the dataset, none of the '{CYAN}{args.cases}{RESET}{RED}' cases belonging to the classes you removed will be available to be used",  flush=True        )
      print( f"{RED}P_C_GENERATE:    ERROR:   (test mode) not halting, but this will likely cause problems",  flush=True      )
      time.sleep(4)
    else:
      print( f"{ORANGE}P_C_GENERATE:    ERROR:   (training mode) the number of cases found and processed ({MIKADO}{directories_processed}{RESET}{ORANGE}) is less than the number  requested ({MIKADO}{cases_required}{RESET}{ORANGE})",  flush=True        ) 
      print( f"{ORANGE}P_C_GENERATE:    ERROR:   (training mode)   possible explanation: if you set '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{ORANGE}' to a number less than the number of classes actually present in the dataset, none of the '{CYAN}{args.cases}{RESET}{ORANGE}' cases belonging to the classes you removed will be available to be used",  flush=True        )
      print( f"{ORANGE}P_C_GENERATE:    ERROR:   (training mode) not halting, but this might cause problems",  flush=True      )
  

  
  if DEBUG>0:
    print( f"{ASPARAGUS}P_C_GENERATE:   INFO:   directories_processed = {MIKADO}{directories_processed:<4d}{RESET}",  flush=True        )   

  if DEBUG>2:
    print( f"\n{RESET}P_C_GENERATE:   INFO:     images_new.shape               = {MIKADO}{images_new.shape}{RESET}",             flush=True       ) 
    print( f"P_C_GENERATE:   INFO:     fnames_new.shape               = {MIKADO}{fnames_new.shape}{RESET}",             flush=True       )
    print( f"P_C_GENERATE:   INFO:     img_labels_new.shape           = {MIKADO}{img_labels_new.shape}{RESET}",         flush=True       )
  if DEBUG>2:
    print( f"P_C_GENERATE:   INFO:     img_labels_new                 = \n{MIKADO}{img_labels_new}{RESET}",             flush=True       )


  if DEBUG>0:
    print( f"P_C_GENERATE:   INFO:    tiles drawn from each of the {MIKADO}{highest_class_number}{RESET} cancer types = {MIKADO}{class_counts}{RESET}",             flush=True       )

  # trim, then convert everything into Torch style tensors

  images_new      = images_new     [0:global_tiles_processed]
  img_labels_new  = img_labels_new [0:global_tiles_processed]
  fnames_new      = fnames_new     [0:global_tiles_processed]
  
  images_new      = torch.Tensor( images_new )
  fnames_new      = torch.Tensor( fnames_new ).long()
  fnames_new.requires_grad_( False )
  img_labels_new  = torch.Tensor( img_labels_new ).long()                                                # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  img_labels_new.requires_grad_( False )

  if DEBUG>1:
    print( "\nP_C_GENERATE:   INFO:   finished converting image data and labels from numpy array to Torch tensor")


  if DEBUG>2:
    print ( f"P_C_GENERATE:   INFO:     img_labels_new                =                               \n{MIKADO}{img_labels_new}{RESET}{CLEAR_LINE}"    )  




  # save numpy array for subsequent use by clustering functions

  if DEBUG>0:  
    print( f"P_C_GENERATE:   INFO:    {COTTON_CANDY}now saving numpy version of image and labels arrays for possible subsequent use by clustering functions{RESET}{CLEAR_LINE}")
    
  fqn =  f"{args.base_dir}/logs/images_new"
  np.save ( fqn, images_new )

  fqn =  f"{args.base_dir}/logs/img_labels_new"
  np.save ( fqn, img_labels_new )




  # save torch tensor as '.pth' file for subsequent loading by dataset function

  fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_{target}.pth"
  
  if DEBUG>8:  
    print( f"P_C_GENERATE:   INFO:    {PINK}now saving as Torch dictionary (this takes a little time){RESET}{CLEAR_LINE}")
  
  torch.save({
      'images':     images_new,
      'fnames':     fnames_new,
      'img_labels': img_labels_new,
  }, fqn )

    
  print( f"P_C_GENERATE:   INFO:    finished saving Torch dictionary to {MAGENTA}{fqn}{RESET}{CLEAR_LINE}" )
  

  return global_tiles_processed
  
  
