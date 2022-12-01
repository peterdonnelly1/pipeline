"""================================================================================================================================================

Generate torch dictionary from pre-processed TCGA image tiles or gene expression vectors
            
================================================================================================================================================"""

import cv2
import os
import re
import sys
import time
import torch
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

from pathlib               import Path
from torchvision           import transforms

from modes.classify.config import classifyConfig

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

from constants  import *

DEBUG   = 1

rows=26
cols=26


def generate( args, class_names, n_samples, total_slides_counted_train, total_slides_counted_test, total_tiles_required_train, total_tiles_required_test, batch_size, highest_class_number, multimode_case_count, unimode_case_matched_count, unimode_case_unmatched_count, 
              unimode_case____image_count, unimode_case____image_test_count, unimode_case____rna_count, unimode_case____rna_test_count, pct_test, n_tiles, top_up_factors_train, top_up_factors_test, tile_size, 
              low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform ):

  DEBUG     = args.debug_level_generate
  LOG_LEVEL = args.log_level

  # DON'T USE args.n_samples or args.batch_size or args.n_tiles or args.gene_data_norm or args.tile_size or args.highest_class_number or args.low_expression_threshold or args.cutoff_percentile since these are job-level lists. 
  # Here we are using one value of each, passed in as per the above parameters
  just_test                    = args.just_test
  data_dir                     = args.data_dir
  data_source                  = args.data_source
  dataset                      = args.dataset
  input_mode                   = args.input_mode
  pretrain                     = args.pretrain
  cases                        = args.cases
  cases_reserved_for_image_rna = args.cases_reserved_for_image_rna
  rna_file_name                = args.rna_file_name
  rna_file_suffix              = args.rna_file_suffix  
  rna_file_reduced_suffix      = args.rna_file_reduced_suffix
  class_numpy_file_name        = args.class_numpy_file_name
  use_autoencoder_output       = args.use_autoencoder_output
  use_unfiltered_data          = args.use_unfiltered_data
  nn_type_img                  = args.nn_type_img




  
  # (0) 'skip generation' option for rna (only). Even though we won't generate the pytorch files, we still need to open the existing .pth file to determine n_samples, n_genes; and possibly also modify batch_size

  if  (input_mode=='rna') & ( args.skip_generation=='True' ) & ( args.cases == 'ALL_ELIGIBLE_CASES') :    

    if just_test!='True':
      fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_rna_train.pth"
    else:
      fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_rna_test.pth"

    if DEBUG>0:
      print( f"{BOLD}{ORANGE}GENERATE:       INFO:  {CYAN}-g{RESET}{BOLD}{ORANGE} flag ({CYAN}SKIP_GENERATION{RESET}{BOLD}{ORANGE}) is set so loading pre-existing pytorch dataset (which MUST exist): {RESET}{MAGENTA}modes/{args.mode}/dataset_rna_train.pth{RESET}",  flush=True )
    try:
      data       = torch.load( fqn )
      genes      = data['genes']
      n_samples  = genes.shape[0]
      n_genes    = genes.shape[2]
    except Exception as e:
      print ( f"{RED}GENERATE:       INFO:  could not load {MAGENTA}{fqn}{RESET}. Disable -X to force it to be regenerated ... can't continue, so halting now [143]{RESET}" )
      if DEBUG>0:
        print ( f"{RED}GENERATE:       INFO:  the exception was: {CYAN}'{e}'{RESET}" )
      sys.exit(0)

    if DEBUG>2:
      print ( f"GENERATE:       INFO:    n_samples      = {MIKADO}{n_samples}{RESET}",   flush=True         )  
      print ( f"GENERATE:       INFO:    n_genes        = {MIKADO}{n_genes}{RESET}",     flush=True         ) 
  
    if n_samples > n_samples:
      print( f"{ORANGE}GENERATE:       WARNG: proposed number of samples {CYAN}N_SAMPLES{RESET}{ORANGE} ({MIKADO}{n_samples}{ORANGE}) is greater than the number of cases processed, 'n_samples' ( = {MIKADO}{n_samples}{RESET}{ORANGE}){RESET}" )
      print( f"{ORANGE}GENERATE:       WARNG: now changing {CYAN}n_samples{ORANGE} to {MIKADO}{n_samples}{RESET}{RESET}" )
      print( f"{ORANGE}GENERATE:       WARNG: explanation: perhaps you specified a flag such as {CYAN}MULTIMODE____TEST{RESET}{ORANGE}, which selects a subset of the available samples, and this subset is smaller that {CYAN}{n_samples}{RESET}{ORANGE}. This is perfectly fine.{RESET}" )
      n_samples = n_samples
  
    if batch_size > n_samples:
      print( f"{ORANGE}GENERATE:       WARNG: proposed batch size ({CYAN}BATCH_SIZE{RESET} = {MIKADO}{batch_size}{RESET}{ORANGE}) is greater than the number of cases available, 'n_samples'  ( = {MIKADO}{n_samples}{RESET}{ORANGE})" )
      print( f"{ORANGE}GENERATE:       WARNG: changing {CYAN}batch_size){CYAN} to {MIKADO}{int(0.2*n_samples)}{RESET}" )
      print( f"{ORANGE}GENERATE:       WARNG: further comment: If you don't like this value of {CYAN}BATCH_SIZE{RESET}{ORANGE}, stop the program and provide a new value in the configuration file {MAGENTA}conf.py{RESET}")
      batch_size = int(0.2*n_samples)


    return ( n_genes, n_samples, batch_size, 0  )
  

  if  (input_mode=='rna') & ( args.skip_generation=='True' ) & ( args.cases != 'ALL_ELIGIBLE_CASES') : 
      print ( f"{BOLD}{ORANGE}GENERATE:       INFO:  {CYAN}-X{RESET} flag (SKIP_GENERATION) is ony allowed if {CYAN}CASES='ALL_ELIGIBLE_CASES'{RESET}. {BOLD}{ORANGE}It will be ignored, and the dataset will be regenerated{RESET}" )  

    

  if DEBUG>6:
    print( "GENERATE:       INFO:   \
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
 
  cfg = classifyConfig( 0,0 )





  # (1) analyse working data directory and save statistics for later use

  if use_unfiltered_data==True:
    rna_suffix = rna_file_suffix[1:]
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
          print( f"GENERATE:       INFO:    \033[58Cdirectory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq file(s) and {MIKADO}{image_file_count:<2d}{RESET} image files(s) and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )
          time.sleep(0.5)       
        else:
          print( f"GENERATE:       INFO:    directory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq files, {MIKADO}{image_file_count:<2d}{RESET} image files and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )







  # (2) process IMAGE data if applicable
  
  if dataset =='cifr':                                                                                     # CIFAR10 is a special case. Pytorch has methods to retrieve cifar and some other benchmakring databases, and stores them in a format that is ready for immediate loading, hence earlier steps like tiling, and also the generation steps that have to be applied to GDC datasets can be skipped
  
    if DEBUG>=0:
      print( f"{ORANGE}GENERATE:       NOTE:    about  to load cifar-10 dataset" )
      
    cifar           = datasets.CIFAR10( root=args.data_source, train=True,  download=True )                # CIFAR10 data is stored as a numpy array, so it has to be converted to a tensor. # MNIST data and labels are stored as a tensor. CIFAR10 data is stored as numpy array. Which was unexpected.
    img_labels_new  = np.asarray(cifar.targets)[0:n_samples]
    fnames_new      = np.zeros_like(img_labels_new)

    images_new      = cifar.data[0:n_samples]

    if ( tile_size<299) & ( (any( 'INCEPT'in item for item in args.nn_type_img ) ) ):

      print( f"{BOLD_ORANGE}GENERATE:       WARN: tile_size = {MIKADO}{tile_size}{BOLD_ORANGE}. However, for '{CYAN}NN_TYPE_IMG{BOLD_ORANGE}={MIKADO}INCEPT4{BOLD_ORANGE}', the tile size must be at least {MIKADO}299x299{RESET}" )
      print( f"{BOLD_ORANGE}GENERATE:       WARN: upsizing the {CYAN}tile_size{BOLD_ORANGE} of all images to {MIKADO}299x299{RESET}{RESET}" )
      
      items = images_new.shape[0]
      images_new_upsized = np.zeros( ( items, 299, 299, 3 ), dtype=np.uint8  )
      
      for i in range ( 0, images_new.shape[0] ):      

        if i%100:
          print ( f"\rGENERATE:       INFO: {i+1} of {items} images have been upsized", end='', flush=True)
        
        item = images_new[i]
        
        xform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299))
            #transforms.RandomRotation((0, 360)),
            #transforms.RandomCrop(cfg.IMG_SIZE),
            #transforms.RandomHorizontalFlip(),
            # ~ transforms.ToTensor()
        ])
        
        images_new_upsized[i]  = xform(item) 
  
      print( "" )
      images_new = images_new_upsized
      tile_size         = 299
      args.tile_size[0] = 299


    images_new  = images_new.swapaxes(1,3)                                                           # it's stored as 50000,32,32,3 whereas we need 50,000,3,32,32 so we swap second and last axes around


    if DEBUG>10:
      print( f"{BOLD_CAMEL }GENERATE:       INFO:         type (images_new        )   = {type (images_new)       }{RESET}"      )
      print( f"{BOLD_CAMEL }GENERATE:       INFO:         type (images_new        )   = {type (images_new)       }{RESET}"      )
      print( f"{CAMEL      }GENERATE:       INFO:         images_new.shape        )   = {images_new.shape        }{RESET}"      )
      print( f"{BOLD_CAMEL }GENERATE:       INFO:         type (img_labels_new    )   = {type (img_labels_new)   }{RESET}"      )
      print( f"{CAMEL      }GENERATE:       INFO:         img_labels_new.shape    )   = {img_labels_new.shape    }{RESET}"      )
      print( f"{BOLD_CAMEL }GENERATE:       INFO:         type (fnames_new        )   = {type (fnames_new)       }{RESET}"      )
      print( f"{CAMEL      }GENERATE:       INFO:         fnames_new.shape        )   = {fnames_new.shape        }{RESET}"      )
    if DEBUG>10:
      np.set_printoptions(formatter={'float': lambda x: "{:>0.3f}".format(x)})
      print( f"{CAMEL      }GENERATE:       INFO:         images_new[0,0,:,:]         = \n{images_new[0,0,:,:]   }{RESET}"      )
      np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
      print( f"{CAMEL      }GENERATE:       INFO:         img_labels_new[0:50]        = {img_labels_new [0:50]   }{RESET}"      )
      print( f"{CAMEL      }GENERATE:       INFO:         fnames_new    [0:50]        = {fnames_new     [0:50]   }{RESET}"      )


    images_new      = torch.Tensor( images_new )
    fnames_new      = torch.Tensor( fnames_new ).long()
    fnames_new.requires_grad_    ( False )
    img_labels_new  = torch.Tensor( img_labels_new ).long()                                                # have to explicity cast as long as torch. Apparently tensor does not automatically pick up type from the numpy array in this case.
    img_labels_new.requires_grad_( False )  
 
    if DEBUG>0:
      print( f"{BOLD_AMETHYST}GENERATE:       INFO:     tensor type (images_new     )   = {type (images_new)       }{RESET}"      )
      print( f"{BOLD_AMETHYST}GENERATE:       INFO:          tensor images_new.size()   = {images_new.size()       }{RESET}"      )
        
    if DEBUG>1:
      print( "\nGENERATE:       INFO:   finished converting image data and labels from numpy array to Torch tensor")
  
  
    for target in [ 'image_train' ]:
          
      # save torch tensors as '.pth' file for subsequent loading by dataset function
      
      fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_{target}.pth"
      
      if DEBUG>0:  
        print( f"GENERATE:       INFO:    {PINK}now saving as Torch dictionary (this can take some time in the case of images){RESET}{CLEAR_LINE}")
      
      torch.save({
          'images':     images_new,
          'fnames':     fnames_new,
          'img_labels': img_labels_new,
      }, fqn )
  
    return ( SUCCESS, SUCCESS, SUCCESS, tile_size )  
    
  
  
  
  elif ( input_mode=='image' ) & ( pretrain!='True' ):

    # (2A)  preliminary bits and pieces

    if DEBUG>2:
      print( f"{ORANGE}GENERATE:       NOTE:    input_mode is '{RESET}{CYAN}{input_mode}{RESET}{ORANGE}', so rna and other data will not be generated{RESET}" )  
      
    if cumulative_png_file_count==0:
      print ( f"{RED}GENERATE:       FATAL:  there are no tile files ('png' files) at all. To generate tiles, run '{CYAN}./do_all.sh -d <cancer type code> -i image -c <CASES SELECTOR>{RESET}{RED}' ... halting now{RESET}", flush=True )                 
      sys.exit(0)         


    #  (2B) perhaps generate test dataset   
      
    if args.just_test=='True':

      if args.cases == 'UNIMODE_CASE':

        target                = 'image_test'
        test_cases            = n_samples                                                                  # for 'just_test', n_samples is the number of cases user wants to be processed (i.e NOT n_samples * pct_test, which would be the case in training mode)
        cases_required        = test_cases 
        tiles_required        = total_tiles_required_test
        top_up_factors        =  top_up_factors_test
        case_designation_flag = args.cases

      elif args.cases == 'MULTIMODE____TEST':
        target                = 'image_test'
        cases_required        = cases_reserved_for_image_rna
        tiles_required        = total_tiles_required_test
        top_up_factors        =  top_up_factors_test
        case_designation_flag = args.cases

      else:
        print ( f"{RED}GENERATE:       FATAL: target {CYAN}{target}{RESET} is not catered for in {CYAN}generate(){RESET} for case {CYAN}{just_test}{RESET}" )
        print ( f"{RED}GENERATE:       FATAL: Cannot continue ... halting now{RESET}" )
        sys.exit(0)

        
      if DEBUG>0:
        print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
        print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) case_designation_flag-------------------------------------------------------------- = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
        print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) n_tiles (this run)----------------------------------------------------------------- = {MIKADO}{n_tiles}{RESET}",                flush=True )
        print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) cases_required -------------------------------------------------------------------- = {MIKADO}{cases_required}{RESET}",         flush=True )

      cases_processed, global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tiles_required, tile_size, top_up_factors )

      if DEBUG>0:
        print ( f"{DULL_WHITE}GENERATE:       INFO:    global_tiles_processed (this run)-------------------------------------------------- = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )



    #  (2Bii)  perhaps generate training dataset
      
    else:

      if args.cases=='UNIMODE_CASE':
        
        #  case_designation_flag for             training set = UNIMODE_CASE____IMAGE
        #  case_designation_flag for in-training test     set = UNIMODE_CASE____IMAGE_TEST
      
        test_cases      = int( n_samples * pct_test )
        training_cases  = n_samples - test_cases
                      
        for target in [ 'image_train', 'image_test' ]:
    
          if target=='image_train':

            cases_required        =  total_slides_counted_train
            tiles_required        =  total_tiles_required_train
            top_up_factors        =  top_up_factors_train
            case_designation_flag =  'UNIMODE_CASE____IMAGE'
            
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) n_samples (this run)............................................................... = {MIKADO}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) n_tiles   (this run)............................................................... = {MIKADO}{n_tiles}{RESET}",                                  flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) cases_required (training cases = int(n_samples * (1 - pct_test ) ) ................ = {MIKADO}{cases_required}{RESET}",                           flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) tiles_required .................................................................... = {MIKADO}{tiles_required}{RESET}",                           flush=True )


          if target=='image_test':

            cases_required        =  total_slides_counted_test
            tiles_required        =  total_tiles_required_test
            top_up_factors        =  top_up_factors_test
            case_designation_flag =  'UNIMODE_CASE____IMAGE_TEST'
            
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) case_designation_flag-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  = {BLEU}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) n_samples (this run)-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   = {BLEU}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) n_tiles   (this run)-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   = {BLEU}{n_tiles}{RESET}",                                  flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) pct_test  (this run)-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   = {BLEU}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) cases_required (test cases = n_samples - training_cases) -  -  -  -  -  -  -  -  -  = {BLEU}{cases_required}{RESET}",                           flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: ({target}) tiles_required-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   = {BLEU}{tiles_required}{RESET}",                           flush=True )

    
          cases_processed, global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tiles_required, tile_size, top_up_factors )

        if DEBUG>0:
          print ( f"{DULL_WHITE}GENERATE:       INFO:    global_tiles_processed  (this run)-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  . = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )




        # (2Bb) case_designation_flag for training set = args.cases
        #       case_designation_flag for test set     = args.cases
        #       both training and test sets will be drawn from the same set of examples
      
      elif args.cases == 'ALL_ELIGIBLE_CASES':

        target                = 'image_train'
        tiles_required        =  total_tiles_required_train
        top_up_factors        =  top_up_factors_train
        cases_required        =  n_samples
        case_designation_flag =  args.cases
        
        if DEBUG>2:
          print ( f"{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset", flush=True )
        if DEBUG>2:
          print ( f"{DULL_WHITE}GENERATE:       INFO:    case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}{CLEAR_LINE}",  flush=True )
          print ( f"{DULL_WHITE}GENERATE:       INFO:    cases_required (this run).......................................................... = {MIKADO}{n_samples}{RESET}{CLEAR_LINE}",              flush=True )
          print ( f"{DULL_WHITE}GENERATE:       INFO:    pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}{CLEAR_LINE}",               flush=True )
  
        cases_processed, global_tiles_processed = generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles, tiles_required, tile_size, top_up_factors )

        if DEBUG>0:
          print ( f"{DULL_WHITE}GENERATE:       INFO:    global_tiles_processed  (this run)................................................. = {MIKADO}{global_tiles_processed}{RESET}{CLEAR_LINE}", flush=True )

    return ( SUCCESS, SUCCESS, SUCCESS, 0)
    











  # (3) process "IMAGE_RNA" data, if applicable, with generation of concatenated embeddings as a preliminary step 

  if ( input_mode=='image_rna' ):

    if DEBUG>0:
      print( f"{CHARTREUSE}GENERATE:       NOTE:  input_mode is '{CYAN}{input_mode}{RESET}{CHARTREUSE}', so image and other data will not be generated{RESET}" )  
    
      
    # (3A) preliminary step: create concatenated image+rna embeddings

    dirs_which_have_matched_image_rna_files      = 0
    dirs_which_dont_have_matched_image_rna_files = 0
    designated_case_count                        = 0    
    
    found_count=0
        
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      # (i) find qualifying cases ( these are: 'has_matched_image_rna_data && designated_case_flag' )
      
      if DEBUG>888:  
        print( f"{DIM_WHITE}GENERATE:       INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
      has_matched_image_rna_data = False
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
                
        try:
          fqn = f"{dir_path}/_image_rna_matched___rna.npy"                                                 # if it has an rna embedding file, it must have both image and rna data
          f = open( fqn, 'r' )
          has_matched_image_rna_data=True
          if DEBUG>6:
            print ( f"{PALE_GREEN}GENERATE:       INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[103C has both matched and rna files (listed above)  \r\033[200C (count= {dirs_which_have_matched_image_rna_files+1}{RESET}{PALE_GREEN})",  flush=True )
          dirs_which_have_matched_image_rna_files+=1
        except Exception:
          if DEBUG>6:
            print ( f"{PALE_RED}GENERATE:       INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_RED} \r\033[103C DOES NOT have both matched and rna files (listed above)  \r\033[200C (count= {dirs_which_dont_have_matched_image_rna_files+1}{RESET}{PALE_RED})",  flush=True )
            dirs_which_dont_have_matched_image_rna_files+=1

        designated_case_flag = False
        if has_matched_image_rna_data==True:
          try:
            fqn = f"{dir_path}/{args.cases}"
            if DEBUG>6:
              print ( f"{PURPLE}GENERATE:       INFO:   fqn = {CYAN}{fqn}{RESET}",  flush=True )
            f = open( fqn, 'r' )
            designated_case_flag=True
            if DEBUG>6:
              print ( f"{PURPLE}GENERATE:       INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PURPLE} \r\033[103C is a designated case \r\033[150C ({CYAN}case flag = {MAGENTA}{args.cases}{RESET}{PURPLE} \r\033[200C (count= {designated_case_count+1}{RESET}{PURPLE})",  flush=True )
            designated_case_count+=1
          except Exception:
            if DEBUG>6:
              print ( f"{PALE_RED}GENERATE:       INFO:   case  {RESET}{CYAN}{dir_path}{RESET}{PALE_RED} \r\033[103C is NOT a designated case \r\033[150C ({CYAN}case = {MAGENTA}{args.cases}{RESET}{PALE_RED})",  flush=True )

        
        #  (ii) make the concatenated embedding
        
        if has_matched_image_rna_data & designated_case_flag:
          
          for f in sorted( files ):
            
            if f.endswith( args.embedding_file_suffix_image ):
              found_count+=1
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image embedding file        =  {ARYLIDE}{f}{RESET}",  flush=True )
              try:
                fqn = f"{dir_path}/{f}"
                image_embedding = np.load ( fqn )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: {e} {RESET}" )
                print ( f"{RED}GENERATE:       FATAL: Explanation: each case must have both an image and an rna embedding. This case ({MAGENTA}{fqn}{RESET}{RED}) does not have an image embedding{RESET}" )
                print ( f"{RED}GENERATE:       FATAL: Cannot continue ... halting now{RESET}" )
                sys.exit(0)

              try:
                fqn = f"{dir_path}/{f}"
                rna_embedding   = np.load ( fqn )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: {e} {RESET}" )
                print ( f"{RED}GENERATE:       FATAL: Explanation: each case must have both an image and an rna embedding. This case ({MAGENTA}{fqn}{RESET}{RED}) does not have an rna   embedding{RESET}" )
                print ( f"{RED}GENERATE:       FATAL: Cannot continue ... halting now{RESET}" )
                sys.exit(0)
              #rna_embedding = np.zeros_like( rna_embedding )                                              # force rna-seq portion of the concatenated embedding to zero for testing purposes
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image_embedding.shape       =  {ARYLIDE}{image_embedding.shape}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}GENERATE:       INFO:   rna_embedding.shape         =  {BLEU}{rna_embedding.shape}{RESET}",       flush=True )
              
              image_rna_embedding = np.concatenate( ( image_embedding, rna_embedding ), axis=0 )
              n_genes             = image_rna_embedding.shape[0]                                           # this will be done for every embedding but that's fine
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image_rna_embedding.shape   =  {BITTER_SWEET}{image_rna_embedding.shape}{RESET}",  flush=True )
              random_name   = f"_{random.randint(10000000, 99999999)}_image_rna_matched___image_rna"
              save_fqn      = f"{dir_path}/{random_name}"
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   saving concatenated embedding {BITTER_SWEET}{save_fqn}{RESET}",  flush=True )
              np.save ( save_fqn, image_rna_embedding )
 
  
    if found_count==0:
      print ( f"{RED}GENERATE:       FATAL: No qualifying cases were found{RESET}" )
      print ( f"{RED}GENERATE:       FATAL:   The test for qualification is this: the case must be flagged as '{CYAN}MULTIMODE____TEST{RESET}{RED}' and it must contain a file named '{MAGENTA}image_rna_matched___rna.npy{RESET}{RED}'{RESET}" )
      print ( f"{RED}GENERATE:       FATAL:   (further information: if a case contains '{MAGENTA}image_rna_matched___rna.npy'{RESET}{RED} it is assumed that it also contains the multiple associated matched image files {MAGENTA}'_XXXXXXX_image_rna_matched___image.npy{RESET}{RED}'){RESET}" )
      print ( f"{RED}GENERATE:       FATAL: Cannot continue ... halting now{RESET}" )
      sys.exit(0)

    if found_count<10:
      print ( f"{ORANGE}GENERATE:       FATAL: Fewer than 10 qualifying cases were found{RESET}" )
      print ( f"{ORANGE}GENERATE:       FATAL: Continuing, but this does not bode well{RESET}" )

    n_samples = designated_case_count

    if DEBUG>0:
      print ( f"{CHARTREUSE}GENERATE:       INFO:   length of image_embedding                       =  {MIKADO}{image_embedding.shape[0]}{RESET}",  flush=True )    
      print ( f"{CHARTREUSE}GENERATE:       INFO:   length of rna_embedding                         =  {MIKADO}{rna_embedding.shape[0]}{RESET}",    flush=True )    
      print ( f"{CHARTREUSE}GENERATE:       INFO:   length of concatenated embeddings               =  {MIKADO}{n_genes}{RESET}",                flush=True )    
      print ( f"{CHARTREUSE}GENERATE:       INFO:   n_samples                                       =  {MIKADO}{n_samples}{RESET}",              flush=True )    



    # (3B) set up numpy data structures to accumulate image_rna data as it is processed 

    # need to know required_number_of_image_rna_files in advance to be able to create numpy array to hold them. Determine using this rule: one concatenated image_rna file (___image_rna.npy) will be created for every existing IMAGE embedding file in a directory that has both an image embedding file (___image.npy)and an rna embedding file (___rna.npy)         

    required_number_of_image_rna_files = n_samples * n_tiles

    if DEBUG>0:
      print ( f"{CHARTREUSE}GENERATE:       INFO:   (hence) required_number_of_image_rna_files      =  {MIKADO}{required_number_of_image_rna_files}{RESET}",   flush=True )


    global_image_rna_files_processed =  0                                                                                         # global count of genes processed
    
    if use_autoencoder_output=='False':
      genes_new      = np.zeros( ( required_number_of_image_rna_files, 1, n_genes                ), dtype=np.float64 )
    fnames_new       = np.zeros( ( required_number_of_image_rna_files                            ), dtype=np.int64   )              
    gnames_new       = np.zeros( ( required_number_of_image_rna_files                            ), dtype=np.uint8   )            # was gene names                                               NOT USED
    rna_labels_new   = np.zeros( ( required_number_of_image_rna_files,                           ), dtype=np.int_    )            # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training



    # (3C) process image_rna data and install in numpy data structures
    
    symlinks_created               = 0
            
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under data_dir
 
      if DEBUG>2:  
        print( f"{DIM_WHITE}GENERATE:       INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
        
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
  
          for f in sorted( files ):              
  
            if  f.endswith( args.embedding_file_suffix_image_rna  ):
              
              image_rna_file         = os.path.join( dir_path, f         )
              label_file             = os.path.join( dir_path, class_numpy_file_name )
  
              # set up the pytorch array
              if DEBUG>8:
                print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {BLEU}{f}{RESET}", flush=True )
              
              if use_autoencoder_output=='False':                                                          # skip gene processing (but do labels and gnames) if we're using autoencoder output 
            
                try:
                  image_rna_embedding = np.load( image_rna_file )
                  if DEBUG>9:
                    print ( f"GENERATE:       INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                    print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                  if DEBUG>999:
                    print ( f"GENERATE:       INFO:         image_rna_embedding             =  '{image_rna_embedding}' "            )
                    print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
                except Exception as e:
                  print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [118]{RESET}" )
                  sys.exit(0)
            
            
                if DEBUG>999:  
                  print( f"GENERATE:       INFO:                     image_rna_embedding = {CYAN}{image_rna_embedding}{RESET}" )              
  
            
                genes_new [global_image_rna_files_processed] =  np.transpose(image_rna_embedding)
                  
                if DEBUG>99:
                  print ( f"GENERATE:       INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                  print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"GENERATE:       INFO:         image_rna_embedding             =  \n'{MIKADO}{np.transpose(image_rna_embedding[1,:])}{RESET}' "      )
                  print ( f"GENERATE:       INFO:         genes_new [{global_image_rna_files_processed}] =  '{CYAN}{genes_new[global_image_rna_files_processed]}{RESET}' ")                       
                
              try:
                label = np.load( label_file)
                if DEBUG>6:
                  print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
                if DEBUG>0:
                  print ( f"{label[0]},", end=', ', flush=True )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: '{e}'{RESET}" )
                print ( f"{PALE_RED}GENERATE:       FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{PALE_RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{PALE_RED}{RESET}" )
                print ( f"{PALE_RED}GENERATE:       FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{PALE_RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{PALE_RED}' to regenerate them{RESET}" ) 
                print ( f"{PALE_RED}GENERATE:       FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{PALE_RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
                print ( f"{PALE_RED}GENERATE:       FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{PALE_RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{PALE_RED}', because without it, no class files can be generated'{RESET}" )                                    
                print ( f"{PALE_RED}GENERATE:       FATAL:  cannot continue - halting now{RESET}" )                 
                sys.exit(0)     
                
              rna_labels_new[global_image_rna_files_processed] =  label[0]
              
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:         label[0][{MIKADO}{global_image_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
            
             # fnames_new [global_image_rna_files_processed  ]  =  image_rna_file_link_id                  # link to folder from which that this image_rna_embedding sample belongs to - passed in as a parameter
              fnames_new [global_image_rna_files_processed  ]  =  777   
            
              if DEBUG>888:
                print ( f"{DIM_WHITE}GENERATE:       INFO:        image_rna_file_link_id = {MIKADO}{image_rna_file_link_id}{RESET}",                          flush=True )
                print ( f"{DIM_WHITE}GENERATE:       INFO:        fnames_new[{MIKADO}{global_image_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_image_rna_files_processed  ]}{RESET}", flush=True )
              
              
              gnames_new [global_image_rna_files_processed]  =  781                                        # any old number. We don't currently use these
            
              if DEBUG>9:
                print ( f"{WHITE}GENERATE:       INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
                time.sleep(.4)  
  
              global_image_rna_files_processed+=1
  
              if DEBUG>99:
                print ( f"{WHITE}GENERATE:       INFO: global_image_rna_files_processed = {MIKADO}{global_image_rna_files_processed}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}GENERATE:       INFO: n_samples                  = {CYAN}{n_samples}{RESET}",                             flush=True )


    # (3D) split into training and test datasets; convert to torch format; save as '.pth' files for subsequent loading by loader/dataset functions
    
    embeddings_available        = len(genes_new)
    test_embeddings_needed      = int( embeddings_available * pct_test )
    training_embeddings_needed  = embeddings_available - test_embeddings_needed 

    print ("\n")
    
    for target in [ 'image_rna_train', 'image_rna_test' ]:

      if DEBUG>0:  
        print( f"{CHARTREUSE}GENERATE:       INFO:  target                  = {MIKADO}{target}{RESET}",                                                                                  flush=True )
        print( f"{CHARTREUSE}GENERATE:       INFO:    embeddings_available  = {MIKADO}{embeddings_available}{RESET}",                                                                    flush=True )
        print( f"{CHARTREUSE}GENERATE:       INFO:    embeddings_needed     = {PINK}{training_embeddings_needed if target=='rna_train' else  test_embeddings_needed}{RESET}",            flush=True )
        
      lo = 0                           if target=='image_rna_train' else training_embeddings_needed        # START at 0                            for training cases;  START at 'training_embeddings_needed' for test cases
      hi = training_embeddings_needed  if target=='image_rna_train' else embeddings_available              # END   at 'training_embeddings_needed' for training cases;  END   at (all the rest go to test cases)

      if DEBUG>0:  
        print( f"{CHARTREUSE}GENERATE:       INFO:    lo ({CYAN}{target}{CHARTREUSE})   = {PINK}{lo}{RESET}",                                                                            flush=True )
        print( f"{CHARTREUSE}GENERATE:       INFO:    hi ({CYAN}{target}{CHARTREUSE})   = {PINK}{hi}{RESET}",                                                                            flush=True )


      genes_new_save       = torch.Tensor( genes_new [lo:hi]  )
      fnames_new_save      = torch.Tensor( fnames_new[lo:hi]  ) 
      fnames_new_save.requires_grad_    ( False )                                                          # no gradients for fnames
      gnames_new_save      = torch.Tensor( gnames_new[lo:hi]  ) 
      gnames_new_save.requires_grad_    ( False )                                                          # no gradients for gnames
      rna_labels_new_save  = torch.Tensor(rna_labels_new[lo:hi])                                           # have to explicity cast as long in torch. Tensor does not automatically pick up type from the numpy array. 
      rna_labels_new_save.requires_grad_( False )                                                          # no gradients for labels
  
      fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_{target}.pth"
      
      if DEBUG>0:  
        print( f"{BOLD}{CHARTREUSE}GENERATE:       INFO:  now saving {CYAN}dataset_{target:7s}{BOLD}{CHARTREUSE} as Torch dictionary{RESET}")
        
      torch.save({
          'genes':      genes_new_save,
          'fnames':     fnames_new_save,
          'gnames':     gnames_new_save, 
          'rna_labels': rna_labels_new_save,           
      }, fqn )
  

 
    return ( n_genes, n_samples, batch_size, 0 )




  
  if ( input_mode=='rna' ):
       

    # (4A) determine 'n_genes' by looking at an (any) rna file, (so that it doesn't have to be manually entered as a user parameter)
    
    # To determine n_genes, (so that it doesn't have to be manually specified), need to examine just ONE of the rna files   
    if DEBUG>0:
      print ( f"GENERATE:       INFO:  about to determine value of 'n_genes'"      )
  
    found_one=False
    for dir_path, dirs, files in os.walk( data_dir ):                                                    # each iteration takes us to a new directory under data_dir
      if not (dir_path==data_dir):                                                                       # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
        
        if check_mapping_file( args, dir_path ) == True:                                                 # doesn't currently do anything becaue custom mapping files not implemented
        
          for f in sorted(files):                                                                        # examine every file in the current directory
            if found_one==True:
              break
            if ( f.endswith( rna_file_suffix[1:]) ):                                                     # have to leave out the asterisk apparently
              if DEBUG>999:
                print (f)     
              rna_file      = os.path.join(dir_path, rna_file_name)
              try:
                rna = np.load( rna_file )
                n_genes=rna.shape[0]
                found_one=True
                if DEBUG>9:
                  print ( f"GENERATE:       INFO:   rna.shape             =  '{MIKADO}{rna.shape}{RESET}' "      )
                if DEBUG>2:
                  print ( f"GENERATE:       INFO:  n_genes (determined)  = {MIKADO}{n_genes}{RESET}"        )
              except Exception as e:
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: error message: '{e}'{RESET}" )
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: explanation: a required rna class file doesn't exist. (Probably none exist){RESET}" )                 
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: did you change from image mode to rna mode but neglect to regenerate the rna files the NN needs for rna mode ? {RESET}" )
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: if so, run '{CYAN}./do_all.sh -d <cancer type code> -i rna {BOLD}{CHARTREUSE}-r True{RESET}{BOLD}{RED}' to generate the rna files{RESET}" )                 
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: when you do this, don't suppress preprocessing or dataset generation (i.e. DON'T use either '{BOLD}{CYAN}-X True{RESET}{BOLD}{RED}' or '{BOLD}{CYAN}-g True{RESET}{BOLD}{RED}')" )                 
                  print ( f"{BOLD}{RED}GENERATE:       FATAL: halting now ...{RESET}" )                 
                  sys.exit(0)

    if found_one==False:                  
      print ( f"{RED}GENERATE:       FATAL: No rna files at all exist in the dataset directory ({MAGENTA}{data_dir}{RESET}{RED})"                                                                          )                 
      print ( f"{PALE_RED}GENERATE:                 Possible explanations:{RESET}"                                                                                                                       )
      print ( f"{PALE_RED}GENERATE:                   (1) The dataset '{CYAN}{args.dataset}{RESET}{PALE_RED}' doesn't have any rna-seq data. It might only have image data{RESET}" )
      print ( f"{PALE_RED}GENERATE:                   (2) Did you change from image mode to rna mode but neglect to run '{CYAN}./do_all.sh{RESET}{PALE_RED}' to generate the files requiPALE_RED for rna mode ? {RESET}" )
      print ( f"{PALE_RED}GENERATE:                       If so, run '{CYAN}./do_all.sh <cancer_type_code> rna{RESET}{PALE_RED}' to generate the rna files{RESET}{PALE_RED}. After that, you will be able to use '{CYAN}./just_run.sh <cancer_type_code> rna{RESET}{PALE_RED}'" )                 
      print ( f"{PALE_RED}GENERATE:               Halting now{RESET}" )                 
      sys.exit(0)




    # (4B)  info and warnings
    
    if ( input_mode=='rna' ):
      if DEBUG>2:
        print( f"GENERATE:       NOTE:  input_mode is '{RESET}{CYAN}{input_mode}{RESET}', so image and other data will not be generated{RESET}" )  

    if use_unfiltered_data==True:
      rna_suffix = rna_file_suffix[1:]
      print( f"{BOLD}{ORANGE}GENERATE:       NOTE:  flag '{CYAN}USE_UNFILTERED_DATA{CYAN}{RESET}{BOLD}{ORANGE}' is set, so all genes listed in file '{CYAN}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}{BOLD}{ORANGE}' are being used{RESET}" )        
    else:
      rna_suffix = rna_file_reduced_suffix
      print( f"{BOLD}{ORANGE}GENERATE:       NOTE: flag '{CYAN}USE_UNFILTERED_DATA{CYAN}{RESET}{BOLD}{ORANGE}' is NOT set, so only the subset of genes specified in '{CYAN}TARGET_GENES_REFERENCE_FILE{RESET}{ORANGE}' = '{CYAN}{args.target_genes_reference_file}{RESET}{ORANGE}' are being used.{RESET}" ) 
      print( f"{ORANGE}GENERATE:       NOTE: Set user parameter {CYAN}'USE_UNFILTERED_DATA'{RESET}{ORANGE} to {MIKADO}True{RESET}{ORANGE} if you wish to use all genes (all genes means all the genes in '{MAGENTA}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}{ORANGE}'){RESET}" ) 
   
   
   
   
    # (4C) set case selection logic variables

    if args.just_test=='True':

      #  (i) generate applicable Test dataset

      if args.cases == 'UNIMODE_CASE':

        target                = 'rna_test'
        cases_required        = n_samples                                                                  # in just_test mode, so pct_test is irelevant
        case_designation_flag = 'UNIMODE_CASE____RNA_TEST'
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO: (just_test) {CYAN}args.cases{RESET} = {MIKADO}{args.cases}{RESET}", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) about to generate .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. {CYAN}{target}{RESET}{DULL_WHITE} dataset", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) case_designation_flag.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) cases_required .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- = {MIKADO}{cases_required}{RESET}",         flush=True )

        global_rna_files_processed_test, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )
  
        if DEBUG>9:
          print ( f"{DULL_WHITE}GENERATE:       INFO:    global_rna_files_processed_test  (this run).-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. = {MIKADO}{global_rna_files_processed_test}{RESET}{CLEAR_LINE}", flush=True )

      elif args.cases == 'MULTIMODE____TEST':

        target                = 'rna_test'
        cases_required        = cases_reserved_for_image_rna
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) case_designation_flag .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . .. = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) cases_required  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . .. = {MIKADO}{cases_required}{RESET}",         flush=True )

        global_rna_files_processed_test, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )
  
        if DEBUG>0:
          print ( f"GENERATE:       INFO:    global_rna_files_processed_test  (this run) .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . . = {MIKADO}{global_rna_files_processed_test}{RESET}{CLEAR_LINE}", flush=True )

      elif args.cases == 'ALL_ELIGIBLE_CASES':

        target                = 'rna_test'
        cases_required        = n_samples
        case_designation_flag = args.cases
        
        if DEBUG>0:
          print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO: (just_test) about to generate {CYAN}{target}{RESET} dataset:", flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) case_designation_flag +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  . .. = {MIKADO}{case_designation_flag}{RESET}",  flush=True )
          print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO: (just_test) cases_required  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  . .. = {MIKADO}{cases_required}{RESET}",         flush=True )

        global_rna_files_processed_test, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )
  
        if DEBUG>0:
          print ( f"GENERATE:       INFO:    global_rna_files_processed_test  (this run) +  .  +  .  +  .  +  .  +  .  +  .  +  .  +  . . = {MIKADO}{global_rna_files_processed_test}{RESET}{CLEAR_LINE}", flush=True )


    else:

      #  (ii)   Generate applicable Training dataset

      if args.cases=='UNIMODE_CASE':
        
        # (a) case_designation_flag for training set = UNIMODE_CASE____RNA
        #     case_designation_flag for test     set = UNIMODE_CASE____RNA_TEST
        
        test_cases      = int( n_samples * pct_test )
        training_cases  = n_samples - test_cases        
                      
        for target in [ 'rna_train', 'rna_test' ]:
    
          if target=='rna_train':
            cases_required        =  training_cases
            case_designation_flag =  'UNIMODE_CASE____RNA'
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_train) case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_train)   n_samples (this run)............................................................. = {MIKADO}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_train)   pct_test  (this run)............................................................. = {MIKADO}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_train)   therefore cases_required (training cases = int(n_samples * (1 - pct_test ) ) .... = {MIKADO}{cases_required}{RESET}",                           flush=True )

            global_rna_files_processed_train, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )

            if DEBUG>0:
              print ( f"{WHITE}GENERATE:       INFO:    global_rna_files_processed_train  (this run)................................................. = {MIKADO}{global_rna_files_processed_train}{RESET}{CLEAR_LINE}", flush=True )


          if target=='rna_test':
            cases_required        =  test_cases
            case_designation_flag =  'UNIMODE_CASE____RNA_TEST'
            if DEBUG>0:
              print ( f"{CLEAR_LINE}{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset:", flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_test)  case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}",                    flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_test)    n_samples (this run)............................................................. = {MIKADO}{n_samples}{RESET}",                                flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_test)    pct_test  (this run)............................................................. = {MIKADO}{pct_test}{RESET}",                                 flush=True )
              print ( f"{CLEAR_LINE}{DULL_WHITE}GENERATE:       INFO:    (rna_test)    therefore cases_required (test cases = n_samples - training_cases) .............. = {MIKADO}{cases_required}{RESET}",                           flush=True )
  
            global_rna_files_processed_test, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )

            if DEBUG>0:
              print ( f"{WHITE}GENERATE:       INFO:    global_rna_files_processed_test  (this run)................................................. = {MIKADO}{global_rna_files_processed_test}{RESET}{CLEAR_LINE}", flush=True )
    
  

      # (b) case_designation_flag for training set = args.cases
      #     case_designation_flag for test set     = args.cases
         
      
      elif args.cases == 'ALL_ELIGIBLE_CASES':

        target                = 'rna_train'
        cases_required        =  n_samples
        case_designation_flag =  args.cases
        
        if DEBUG>2:
          print ( f"{WHITE}GENERATE:       INFO:  about to generate {CYAN}{target}{RESET} dataset", flush=True )
        if DEBUG>2:
          print ( f"{DULL_WHITE}GENERATE:       INFO:    case_designation_flag.............................................................. = {MIKADO}{case_designation_flag}{RESET}{CLEAR_LINE}",  flush=True )
          print ( f"{DULL_WHITE}GENERATE:       INFO:    cases_required (this run).......................................................... = {MIKADO}{n_samples}{RESET}{CLEAR_LINE}",              flush=True )
          print ( f"{DULL_WHITE}GENERATE:       INFO:    pct_test  (this run)............................................................... = {MIKADO}{pct_test}{RESET}{CLEAR_LINE}",               flush=True )


        class_counts = np.zeros( highest_class_number+1, dtype=np.int )        
        global_rna_files_processed_train, n_genes = generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output )


        if DEBUG>0:
          print ( f"{WHITE}GENERATE:       INFO:  global_rna_files_processed_train  (this run)............................................. = {MIKADO}{global_rna_files_processed_train}{RESET}{CLEAR_LINE}", flush=True )


    # ~ if args.n_samples[0] > global_rna_files_processed_test:
      # ~ print( f"{ORANGE}GENERATE:       WARNG: proposed number of samples {CYAN}N_SAMPLES{RESET}{ORANGE} ({MIKADO}{args.n_samples[0]}{ORANGE}) is greater than the number of cases processed, 'global_rna_files_processed_test' ( = {MIKADO}{global_rna_files_processed_test}{RESET}{ORANGE}){RESET}" )
      # ~ print( f"{ORANGE}GENERATE:       WARNG: now changing {CYAN}args.n_samples[0]{ORANGE} to {MIKADO}{global_rna_files_processed_test}{RESET}{RESET}" )
      # ~ print( f"{ORANGE}GENERATE:       WARNG: explanation: perhaps you specified a flag such as {CYAN}MULTIMODE____TEST{RESET}{ORANGE}, which selects a subset of the available samples, and this subset is smaller that {CYAN}{n_samples}{RESET}{ORANGE}. This is perfectly fine.{RESET}" )
      # ~ args.n_samples[0] = global_rna_files_processed_test
    if args.batch_size[0] > global_rna_files_processed_test:
      print( f"{ORANGE}GENERATE:       WARNG: proposed batch size ({CYAN}BATCH_SIZE{RESET} = {MIKADO}{args.batch_size[0]}{RESET}{ORANGE}) is greater than the number of cases available, 'global_rna_files_processed_test'  ( = {MIKADO}{global_rna_files_processed_test}{RESET}{ORANGE})" )
      print( f"{ORANGE}GENERATE:       WARNG: changing {CYAN}args.batch_size[0]){CYAN} to {MIKADO}{global_rna_files_processed_test}{RESET}" )
      print( f"{ORANGE}GENERATE:       WARNG: further comment: If you don't like this value of {CYAN}BATCH_SIZE{RESET}{ORANGE}, stop the program and provide a new value in the configuration file {MAGENTA}conf.py{RESET}")
      batch_size = int(global_rna_files_processed_test)

  
  
  
    # (4D) RETURN
  
    return ( n_genes, n_samples, batch_size, 0 )




  ########################################################################################################################################################################################################
  #
  #  These are all the valid cases for rna:
  #       
  #  user flag:
  # -c ALL_ELIGIBLE_CASES                      <<< Largest possible set. For use in unimode experiments only (doesn't set aside any test cases for multimode):
  # -c UNIMODE_CASE                            <<< Largest set that can be used in multimode experiments (because it uses ummatched cases in unimode runs):
  # -c UNIMODE_CASE____MATCHED                 <<< Combination to use when testing the thesis (uses only matched cases for unimode runs)                                                   <<< not currently implemented but needs to be
  # -c MULTIMODE____TEST                       <<< Cases exclusively set aside for MULTIMODE testing. These cases are guaranteed to have never been seen during UNIMODE testing
  #
  #  What to generate as the TRAINING set:
  #  If  -c = ... 
  #    ALL_ELIGIBLE_CASES                      then grab these cases:
  #    if -i rna:
  #       UNIMODE_CASE                         then grab these cases: UNIMODE_CASE____RNA  &! MULTIMODE____TEST                                 <<< currently catered for
  #       UNIMODE_CASE____MATCHED              then grab these cases: <tbd>                                                                     <<< not currently implemented. Uses only matched cases for unimode runs 
  #    if -i rna:
  #       UNIMODE_CASE                         then grab these cases: UNIMODE_CASE____RNA_FLAG  &! MULTIMODE____TEST                            <<< currently catered for
  #       UNIMODE_CASE____MATCHED              then grab these cases: <tbd>                                                                     <<< not currently implemented. Uses only matched cases for unimode runs 
  #       MULTIMODE____TEST          N/A                                                                                                        <<< Never used in training
  #
  #  What to generate as the TEST set:
  #  If -c = ...
  #    ALL_ELIGIBLE_CASES                      then grab these cases:
  #    if -i rna:
  #       UNIMODE_CASE                         then grab these cases: UNIMODE_CASE____RNA_TEST                                                  <<< currently catered for
  #       UNIMODE_CASE____MATCHED              then grab these cases:  <tbd>                                                                    <<< not currently implemented. Uses only matched cases for unimode runs
  #    if -i rna:
  #       UNIMODE_CASE                         then grab these cases: UNIMODE_CASE____RNA_TEST_FLAG                                             <<< currently catered for
  #       UNIMODE_CASE____MATCHED              then grab these cases: <tbd>                                                                     <<< not currently implemented. Uses only matched cases for unimode runs
  #    MULTIMODE____TEST                       then grab these cases: MULTIMODE____TEST                                                         <<< the set that's exclusively reserved for multimode testing (always matched)
  #
  #
  #  ------------------------------------------+-----------------------------------------------------------------------------------------------------+----------------------------------------------------
  #                 User Flag                  |                                             Training                                                |                      Test                         
  #  ------------------------------------------+-----------------------------------------------------------------------------------------------------+----------------------------------------------------
  #                                count:      |               1 - (pct_test * n_samples)       |               (pct_test * n_samples)               |         cases_reserved_for_image_rna
  #  ------------------------------------------+------------------------------------------------+----------------------------------------------------+----------------------------------------------------
  #                                            |                                                |                                                    |
  #  -c ALL_ELIGIBLE_CASES                     |         !MULTIMODE____TEST                     |      UNIMODE_CASE____RNA_TEST                      |                     -
  #                                            |                                                |                                                    |
  #  -c UNIMODE_CASE                           |          UNIMODE_CASE____RNA                   |      UNIMODE_CASE____RNA_TEST                      |         MULTIMODE____TEST
  #                                            |          UNIMODE_CASE____RNA_FLAG              |      UNIMODE_CASE____RNA_TEST_FLAG                 |         MULTIMODE____TEST
  #                                            |                                                |                                                    |
  #  -c UNIMODE_CASE____MATCHED                |                                                |                                                    |         MULTIMODE____TEST
  #                                            |                                                |                                                    |
  #  ------------------------------------------+------------------------------------------------+----------------------------------------------------+----------------------------------------------------
  #  -c MULTIMODE____TEST                      |                                                |                                                    |         MULTIMODE____TEST
  #  -------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------------------------
def generate_rna_dataset ( args, class_names, target, cases_required, highest_class_number, case_designation_flag, n_genes, low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform, use_autoencoder_output  ):

  cases_required = cases_required if cases_required>0 else 1                                               # zero cases will cause a crash 
  
  
  if use_autoencoder_output=='False':
    genes_new           = np.zeros( ( cases_required, 1, n_genes   ), dtype=np.float64 )
  fnames_new          = np.zeros( ( cases_required               ), dtype=np.int64   )              
  gnames_new          = np.zeros( ( cases_required               ), dtype=np.uint8   )        # was gene names                                               NOT USED
  rna_labels_new      = np.zeros( ( cases_required,              ), dtype=np.int_    )        # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training
  rna_files_processed =  0                                                                                 # count of rna files processed


  if DEBUG>2:
    print( f"GENERATE:       INFO:     genes_new.shape                = {PINK}{genes_new.shape}{RESET}",              flush=True       ) 
    print( f"GENERATE:       INFO:     rna_labels_new.shape           = {PINK}{rna_labels_new.shape}{RESET}",         flush=True       ) 
    print( f"GENERATE:       INFO:     fnames_new.shape               = {PINK}{fnames_new.shape}{RESET}",             flush=True       )


  global_cases_processed  = 0
  directories_processed   = 0
  sufficient_cases_found  = False

  
  # (6A) select and process applicable cases (case_designation_flag)

  not_designated_case_count  = 0
  designated_case_count      = 0
  
  for dir_path, dirs, files in os.walk( args.data_dir ):                                                   # each iteration takes us to a new directory under data_dir

    if DEBUG>888:  
      print( f"{DIM_WHITE}GENERATE:       INFO:   now processing case (directory) {CYAN}{dir_path}{RESET}" )
        
    if not (dir_path==args.data_dir):                                                                      # the top level directory (dataset) has be skipped because it only contains sub-directories, not data

      #  (1) is it a one of the cases we're looking for ?  
        
      use_this_case_flag=False
      try:
        fqn = f"{dir_path}/{case_designation_flag}"        
        f = open( fqn, 'r' )
        use_this_case_flag=True
        if DEBUG>2:
          print ( f"\n{GREEN}GENERATE:       INFO:   case \r\033[55C'{COTTON_CANDY}{dir_path}{RESET}{GREEN}' \r\033[130C is     a case flagged as '{CYAN}{case_designation_flag}{RESET}{GREEN}' - - including{RESET}{CLEAR_LINE}",  flush=True )
      except Exception:
        not_designated_case_count+=1
        if DEBUG>4:
          print ( f"{RED}GENERATE:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED} \r\033[130C is not a case flagged as '{CYAN}{case_designation_flag}{RESET}{RED}' - - skipping{RESET}{CLEAR_LINE}",  flush=True )


      if ( use_this_case_flag==True ) | ( args.cases=='ALL_ELIGIBLE_CASES' ):

        for f in sorted( files ):
                                   
          if DEBUG>999:
            print ( f"{DIM_WHITE}GENERATE:       INFO:  rna_suffix                   = {MIKADO}{rna_suffix}{RESET}", flush=True )
            print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {MAGENTA}{f}{RESET}",         flush=True )            
                            
                                  
          # (i) Make and store a  softlink based on an integer reference to the case id for later use so that DENSE will later know where to save the rna-seq embeddings (if this option is enabled)
            
          if  f == args.rna_file_name:                                                                        # always 'rna.npy'

            fqcd                      = f"{dir_path}"
            parent_dir                = os.path.split(fqcd)[1]
            no_special_chars_version  = re.sub('[^A-Za-z0-9]+', '', parent_dir).lstrip()
            final_chars               = no_special_chars_version[-6:]
            int_version               = int( final_chars, 16)
                
            if DEBUG>5:
              print (f"GENERATE:       INFO:              fully qualified case directory = '{MAGENTA}{fqcd}{RESET}'" )
              print (f"GENERATE:       INFO:                                    dir_path = '{MAGENTA}{dir_path}{RESET}'" )
              print (f"GENERATE:       INFO:                                  parent_dir = '{MAGENTA}{parent_dir}{RESET}'" )
              print (f"GENERATE:       INFO:                    no_special_chars_version = '{MAGENTA}{no_special_chars_version}{RESET}'" )
              print (f"GENERATE:       INFO:                                 final_chars = '{MAGENTA}{final_chars}{RESET}'" )
              print (f"GENERATE:       INFO:                                 hex_version = '{MAGENTA}{int_version}{RESET}'" )
      
            rna_file_link_id   = int_version
            rna_file_link_name = f"{rna_file_link_id:d}"
      
            fqln = f"{args.data_dir}/{rna_file_link_name}.fqln"                                                  # name for the link
            try:
              os.symlink( fqcd, fqln)                                                                            # make the link
            except Exception as e:
              if DEBUG>2:
                print ( f"{ORANGE}GENERATE:       NOTE:  Link already exists{RESET}" )
              else:
                pass
      
            if DEBUG>2:
              print( f"GENERATE:       INFO:                            rna_file_link_id =  {MAGENTA}{rna_file_link_id}{RESET}" )
              print( f"GENERATE:       INFO:                          rna_file_link_name = '{MAGENTA}{rna_file_link_name}{RESET}'" )
              print( f"GENERATE:       INFO:                                        fqln = '{MAGENTA}{fqln}{RESET}'" )


            
            # ~ rna_file_link_id   = random.randint(1000000, 9999999)                                          # generate random string to use for the softlink to the file name (can't have strings in tensors)
            # ~ rna_file_link_name = f"{rna_file_link_id:d}"
      
            # ~ fqcd = f"{dir_path}"                                                                           # fully qualified case directory
            # ~ fqln = f"{args.data_dir}/{rna_file_link_name}.fqln"                                            # fully qualified link name
            try:
              os.symlink( fqcd, fqln)                                                                      # make a link from fqln to fqcd
              if DEBUG>2:
                print ( f"GENERATE:       INFO:       softlink (fqln) {MAGENTA}{fqln}{RESET} \r\033[93C and target (fqcd) = {MAGENTA}{fqcd}{RESET}" )
            except Exception as e:
              if DEBUG>2:
                print ( f"{ORANGE}GENERATE:       INFO:       softlink (fqln) {MAGENTA}{fqln}{RESET}{ORANGE} \r\033[93C to target (fqcd) = {MAGENTA}{fqcd}{RESET}{ORANGE} \r\033[185C already exists, which may well be fine and intended{RESET}" )


            # (ii) Process the rna-seq file
            
            rna_file      = os.path.join( dir_path, args.rna_file_name         )
            label_file    = os.path.join( dir_path, args.class_numpy_file_name )

            # set up the pytorch array
          
            if DEBUG>8:
              print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {BLEU}{f}{RESET}", flush=True )
            
            if use_autoencoder_output=='False':                                                                      # Skip gene processing (but do labels and gnames) if we're using autoencoder output. We'll LATER load and use ae output file as genes_new rather than process raw rna-seq data 
          
              try:
                rna = np.load( rna_file )
                if DEBUG>9:
                  print ( f"GENERATE:       INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                  print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"GENERATE:       INFO:         rna             =  '{rna}' "            )
                  print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [2118]{RESET}" )
                sys.exit(0)
          
          
              if DEBUG>999:  
                print( f"GENERATE:       INFO:                     rna = {CYAN}{rna}{RESET}" )              
              
              rna[np.abs(rna) < 1] = 0                                                                               # set all the values lower than 1 to be 0
              
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
              elif gene_data_transform=='RANKED':
                print ( f"{BOLD}{ORANGE}GENERATE:       INFO: {CYAN}RANKED{RESET}{BOLD}{ORANGE} data transformation has been selected. Note that ranking the gene vectors can take several minutes{RESET}")  
                rna = np.array([ el+random.uniform(0.0001,0.0003) for el in rna ])                       # to make sure no two elements are precisely the same
                if DEBUG>99:
                  # ~ print ( f"GENERATE:       INFO:         rna.shape              =  '{MIKADO}{rna.shape}{RESET}' "                                    )  
                  np.set_printoptions(formatter={'float': lambda x: "{:4.2e}".format(x)})
                  print ( f"GENERATE:       INFO:         rna                    =  '{BLEU}{np.transpose(rna[0:23])}{RESET}' "                        )  
                temp                  = rna.argsort(axis=0).squeeze()
                if DEBUG>99:
                  np.set_printoptions(formatter={'int': lambda x: "{:>8d}".format(x)})
                  print ( f"GENERATE:       INFO:         temp                   =  '{COQUELICOT}{np.transpose(temp[0:23])}{RESET}' "       ) 
                transformed_rna       = np.zeros_like(temp)
                transformed_rna[temp] = np.arange( len(rna) )  
                if DEBUG>99:
                  np.set_printoptions(formatter={'int': lambda x: "{:>8d}".format(x)})
                  print ( f"GENERATE:       INFO:         transformed_rna        =  '{AMETHYST}{np.transpose(transformed_rna[0:23])}{RESET}' "       ) 
              else:
                print( f"{RED}GENERATE:      FATAL:        no such gene data transformation as: {gene_data_transform[0:10]} ... halting now[184]{RESET}" )
                sys.exit(0) 
          
              if gene_data_norm=='NONE':
                normalized_rna =  transformed_rna
              elif gene_data_norm=='JUST_SCALE':
                normalized_rna = transformed_rna / np.std(transformed_rna)   
              elif gene_data_norm=='GAUSSIAN':
                normalized_rna = ( transformed_rna - np.mean(transformed_rna) ) / np.std(transformed_rna)                                             
              else:
                print( f"{RED}GENERATE:      FATAL:        no such gene normalization mode as: {gene_data_norm} ... halting now[378]{RESET}" )  
                sys.exit(0)       
          
              genes_new [rna_files_processed] =  np.transpose(normalized_rna)               
                
              if DEBUG>99:
                print ( f"GENERATE:       INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"GENERATE:       INFO:         rna             =  \n'{MIKADO}{np.transpose(rna[1,:])}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new [{rna_files_processed}] =  '{CYAN}{genes_new[rna_files_processed]}{RESET}' ")                       
              
            try:
              label = np.load( label_file)
              if DEBUG>99:
                print ( "GENERATE:       INFO:         label.shape =  \"{:}\"".format(  label.shape) )
                print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
              if DEBUG>2:
                print ( f"{label[0]},", end='', flush=True )
              if label[0]>highest_class_number:
                if DEBUG>0:
                  print ( f"{PALE_ORANGE}GENERATE:       INFO:    {MAGENTA}{os.path.basename(os.path.normpath(dir_path))}{RESET}{PALE_ORANGE} \r\033[66C <<< this case's class label (subtype) ({MIKADO}{label[0]:2d}{RESET}{PALE_ORANGE}) is greater than {CYAN}HIGHEST_CLASS_NUMBER{RESET}{PALE_ORANGE} ({MIKADO}{highest_class_number:2d}{RESET}{PALE_ORANGE}) so it won't be used {RESET}")
                break
            except Exception as e:
              print ( f"{RED}GENERATE:       FATAL: '{e}'{RESET}" )
              print ( f"{RED}GENERATE:       FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{RED}{RESET}" )
              print ( f"{RED}GENERATE:       FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{RED}' to regenerate them{RESET}" ) 
              print ( f"{RED}GENERATE:       FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> -r True{RESET}{RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
              print ( f"{RED}GENERATE:       FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{RED}', because without it, no class files can be generated'{RESET}" )                                    
              print ( f"{RED}GENERATE:       FATAL:  cannot continue - halting now{RESET}" )                 
              sys.exit(0)     


              
            rna_labels_new[rna_files_processed] =  label[0]
            #rna_labels_new[rna_files_processed] =  random.randint(0,5)                        ################### swap truth labels to random numbers for testing purposes
            
            if DEBUG>777:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_labels_new[{MIKADO}{rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
          
          
            fnames_new [rna_files_processed  ]  =  rna_file_link_id                               # link to folder from which that this rna sample belongs to - passed in as a parameter
          
            if DEBUG>888:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_file_link_id = {MIKADO}{rna_file_link_id}{RESET}",                          flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO:        fnames_new[{MIKADO}{rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [rna_files_processed  ]}{RESET}", flush=True )
            
            
            gnames_new [rna_files_processed]  =  443                                              # Any old number. We don't currently use these
          
            if DEBUG>888:
              print ( f"{WHITE}GENERATE:       INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
              time.sleep(.4)  

            rna_files_processed+=1

            if DEBUG>88:
              print ( f"{WHITE}GENERATE:       INFO: rna_files_processed = {MIKADO}{rna_files_processed}{RESET}",  flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO: cases_required                  = {CYAN}{cases_required}{RESET}",               flush=True )
 
    if  rna_files_processed>=cases_required:
      break



  # Maybe plot some statstics relating to the variability in the genes' expression across the examples
    
  if DEBUG>2:
    
    points_to_plot = 200
    np.set_printoptions(formatter={'float': lambda x: f"{x:>14.0f}"})  
      
    print ( f"{WHITE}GENERATE:       INFO: FPKM-UQ for first 10 genes and first 50 examples   = \n{MIKADO}{np.squeeze(genes_new)[0:50,0:10]}{RESET}",  flush=True )  
  

    # Plot the means of some genes across all examples
  
    mean = np.mean( genes_new, axis=0 )
     
    
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: first 10 means               = {MIKADO}{(np.squeeze(mean))[0:10]}{RESET}",            flush=True )
    if DEBUG>2:
      print ( f"{WHITE}GENERATE:       INFO: np.squeeze(mean).shape       = {MIKADO}{np.squeeze(mean).shape}{RESET}",              flush=True )
  
    fig, ax = plt.subplots( figsize=( 20, 30 ) )
  
    title = f"mean values of FPKM-UQ across all examples for first {points_to_plot} genes"
    plt.title   (title,                                                         fontsize=16 )
    plt.xlabel  ("gene",                                                        fontsize=14 )
    plt.ylabel  ("mean",                                                        fontsize=14 )
    plt.tick_params (axis='x', labelsize=8,   labelcolor='black')
    plt.tick_params (axis='y', labelsize=14,  labelcolor='black')
    plt.xticks  ( rotation=90 )
    plt.yscale('log')
    bar_plot = plt.bar(  x = np.arange(points_to_plot), height=(np.squeeze(mean))[0:points_to_plot] )
    plt.show()

    # ~ writer.add_figure('mean_FPKM-UQ_across_all_examples_for_first_few_genes', fig, 22 )

    # Plot the variance of some genes across all examples
    
    variance = np.var( genes_new, axis=0 )
       
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: first 10 variances           = {MIKADO}{(np.squeeze(variance))[0:10]}{RESET}",            flush=True )
    if DEBUG>2:
      print ( f"{WHITE}GENERATE:       INFO: np.squeeze(variance).shape   = {MIKADO}{np.squeeze(variance).shape}{RESET}",              flush=True )
    
    fig, ax = plt.subplots( figsize=( 20, 30 ) )
    title = f"Variance of FPKM-UQ across all examples for first {points_to_plot} genes"
    plt.title   (title,                                                         fontsize=16 )
    plt.xlabel  ("gene",                                                        fontsize=14 )
    plt.ylabel  ("variance",                                                    fontsize=14 )
    plt.tick_params (axis='x', labelsize=8,   labelcolor='black')
    plt.tick_params (axis='y', labelsize=14,  labelcolor='black')
    plt.xticks  ( rotation=90 )
  
    plt.yscale('log')
    bar_plot = plt.bar(  x = np.arange(points_to_plot) , height=(np.squeeze(variance))[0:points_to_plot] )
  
    plt.show()
    
    
    # Plot the standard deviation of some genes across all examples
    
    std = np.std( genes_new, axis=0 )
       
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: first 10 std's               = {MIKADO}{(np.squeeze(std))[0:10]}{RESET}",            flush=True )
    if DEBUG>2:
      print ( f"{WHITE}GENERATE:       INFO: np.squeeze(std).shape        = {MIKADO}{np.squeeze(std).shape}{RESET}",              flush=True )
    
    fig, ax = plt.subplots( figsize=( 20, 30 ) )
    title = f"standard deviation of FPKM-UQ across all examples for first {points_to_plot} genes"
    plt.title   (title,                                                         fontsize=16 )
    plt.xlabel  ("gene",                                                        fontsize=14 )
    plt.ylabel  ("standard deviation",                                          fontsize=14 )
    plt.tick_params (axis='x', labelsize=8,   labelcolor='black')
    plt.tick_params (axis='y', labelsize=14,  labelcolor='black')
    plt.xticks  ( rotation=90 )
  
    plt.yscale('log')
    bar_plot = plt.bar(  x = np.arange(points_to_plot) , height=(np.squeeze(std))[0:points_to_plot] )
  
    plt.show()
    
    
    # tiny var
    
    variance = np.var( genes_new, axis=0 )
    tiny_var = variance[ variance<1 ]
    
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: number of genes with variance less than 1     = {MIKADO}{tiny_var.shape[0]}{RESET}",              flush=True )

    # zero var
    
    variance = np.var( genes_new, axis=0 )
    zero_var = variance[ variance==0 ]
    
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: number of genes with variance exactly   0     = {MIKADO}{zero_var.shape[0]}{RESET}",              flush=True )
    
    # low var
    
    low_variance_threshold = 2000
    variance = np.var( genes_new, axis=0 )
    low_var = variance[ variance<low_variance_threshold ]
    low_var = low_var [ low_var>0    ]
    
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: number  of genes with variance less than {MIKADO}{low_variance_threshold}{RESET}  = {ORANGE}{low_var.shape[0]}{RESET}",              flush=True )
    if DEBUG>0:
      print ( f"{WHITE}GENERATE:       INFO: FPKM-UQ of genes with variance less than {MIKADO}{low_variance_threshold}{RESET} = \n{ORANGE}{low_var}{RESET}",                    flush=True )    
  
  
  

  # (6B) Maybe remove genes with low rna-exp values from all cases
  
  if low_expression_threshold>0:
    
    if DEBUG>0:          
      print ( f"GENERATE:       INFO:{BOLD}{ORANGE}  positive values of {CYAN}COV_THRESHOLD{RESET}{BOLD}{ORANGE} and {BOLD}{CYAN}CUTOFF_PERCENTILE{RESET}{BOLD}{ORANGE} have been set. Removing genes where {BOLD}{MIKADO}{cutoff_percentile}{RESET}{BOLD}{ORANGE}% of cases have FPKM-UQ values less than <{BOLD}{MIKADO}{low_expression_threshold}{RESET}{BOLD}{ORANGE} for all samples{RESET}") 
    if DEBUG>0:
      print( f"GENERATE:       INFO:    {BLEU}genes_new.shape (before)      = {MIKADO}{genes_new.shape}{RESET}",    flush=True )
    if DEBUG>99:        
      print( f"GENERATE:       INFO:        {COQUELICOT}genes_new               = \n{MIKADO}{genes_new}{RESET}",        flush=True )
    if DEBUG>99:
      print( f"GENERATE:       INFO:        about to squeeze genes_new",                                                flush=True ) 
    genes_new = genes_new.squeeze()
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {GREEN}genes_new.shape               = {MIKADO}{genes_new.shape}{RESET}",   flush=True )
    if DEBUG>99:
      print( f"GENERATE:       INFO:        about to calculate   percentiles for each column (gene)",                   flush=True )            
    percentiles  = np.percentile (   np.abs(genes_new), cutoff_percentile, axis=0  )                       # row vector "90% of values lie above ..."
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {PINK}percentiles                   = {MIKADO}{percentiles}{RESET}",        flush=True )        
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {PINK}percentiles.shape             = {MIKADO}{percentiles.shape}{RESET}",  flush=True )        
      print( f"GENERATE:       INFO:        about to apply COV_THRESHOLD to filter out genes that aren't very expressive across all samples (genes whose {MIKADO}{cutoff_percentile}%{RESET} percentile is less than the user provided {CYAN}COV_THRESHOLD{RESET} = {MIKADO}{low_expression_threshold}{RESET})", flush=True )    
    logical_mask      = np.array(  [ ( percentiles ) > low_expression_threshold ]  )                                  # filter out genes that aren't very expressive across all samples
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {PINK}logical_mask.shape            = {MIKADO}{logical_mask.shape}{RESET}", flush=True )
    if DEBUG>99:
      print( f"GENERATE:       INFO:        about to convert logical mask into a integer mask",                         flush=True )          
    integer_mask      = np.squeeze(      logical_mask.astype(int)                  )                       # change type from Boolean to Integer values (0,1) so we can use it as a mask
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {PINK}integer_mask.shape            = {MIKADO}{integer_mask.shape}{RESET}", flush=True )
    if DEBUG>99:                                                                                            # make sure that there are at least SOME non-zero values in the mask or else we'll make an empty matrix in subsequent steps
      print( f"GENERATE:       INFO:        {PINK}integer_mask          = \n{MIKADO}{integer_mask}{RESET}",             flush=True )      
    if np.sum( integer_mask, axis=0 )==0:
      print( f"{RED}GENERATE:        FATAL:    the value provided for COV_THRESHOLD ({MIKADO}{low_expression_threshold}{RESET}{RED}) would filter out {UNDER}every{RESET}{RED} gene -- try a smaller vallue.  Exiting now [755]{RESET}" )
      sys.exit(0)
    non_zero_indices  = np.nonzero (   integer_mask  )                                                     # make a vector of indices corresponding to non-zero values in the mask 

    if DEBUG>99:
      print( f"GENERATE:       INFO:        about to remove all columns corresponding to low correlation genes" ) 
    genes_new = np.take ( genes_new,   non_zero_indices, axis=1                    )                       # take columns corresponding to the indices (i.e. delete the others)
    genes_new = np.squeeze( genes_new )                                                                    # get rid of the extra dimension that for some reason is created in the last step                                                         # convert to numpy, as matplotlib can't use np arrays
    if DEBUG>99:
      print( f"GENERATE:       INFO:        {MIKADO}{logical_mask.shape[1]-genes_new.shape[1]:,}{RESET}{PINK} of {MIKADO}{logical_mask.shape[1]:,}{RESET} {PINK}genes have been removed from consideration{RESET}" ) 

    
    if DEBUG>99:
      print( f"GENERATE:       INFO:{AZURE}        genes_new.shape (after)      = {MIKADO}{genes_new.shape}{RESET}",   flush=True )
    if DEBUG>99:           
      print( f"GENERATE:       INFO:{AZURE}        genes_new[0:rows,0:cols] (after)            = \n{MIKADO}{genes_new[0:rows,0:cols]}{RESET}"       )

    if DEBUG>99:
      print( f"GENERATE:       INFO:        about to expand dims to reinstate the original shape of genes_new" ) 
    genes_new = np.expand_dims(genes_new, axis=1)
    if DEBUG>0:
      print( f"GENERATE:       INFO:{AMETHYST}    genes_new.shape (after)       = {MIKADO}{genes_new.shape}{RESET}", flush=True )

    n_genes = genes_new.shape[2]



  # (6C) convert to torch format

  # (i) convert the genes component to torch format
  
  # If we are in autoencoder mode, ignore the 'genes_new' we just created; and instead substitute the saved feature file (the result of a previous autoencoder run)
  if use_autoencoder_output=='True':                                                                       # then we already have them in Torch format, in the ae feature file, which we now load

    fpath = '%s/ae_output_features.pt' % args.log_dir                                                      # this MUST exist
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}GENERATE:       INFO:  about to load autoencoder generated feature file from {MAGENTA}{fpath}{RESET}", flush=True )
    try:
      genes_new    = torch.load( fpath )
      genes_new    = genes_new.unsqueeze(1)                                                                # add a dimension to make it compatible with existing (non-autoencoder) code
      n_genes      = genes_new.shape[2]                                                                    # i.e. number of gene-like-features from the dimensionality reduced output of the autoencoder
      args.n_genes = n_genes
      if DEBUG>0:
        print ( f"GENERATE:       INFO:    genes_new.size         = {MIKADO}{genes_new.size()}{RESET}"      ) 
        print ( f"GENERATE:       INFO:    n_genes  (determined)  = {MIKADO}{n_genes}{RESET}"               )               
      if DEBUG>0:   
        print( f"{BRIGHT_GREEN}GENERATE:       INFO:  autoencoder feature file successfully loaded{RESET}"  )          
    except Exception as e:
      print ( f"{RED}GENERATE:       INFO:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      if DEBUG>0:
        print ( f"{RED}GENERATE:       INFO:  the exception was: {CYAN}'{e}'{RESET}" )
      sys.exit(0)

  else:
    
    if DEBUG>99:
      print ( f"GENERATE:       INFO:    type(genes_new)         = {MIKADO}{type(genes_new)}{RESET}"      ) 
      print ( f"GENERATE:       INFO:    genes_new.shape         = {MIKADO}{genes_new.shape}{RESET}"      ) 

    genes_new    = torch.Tensor( genes_new )
  
  
  # (ii) convert the other component to torch format
  
  gnames_new      = torch.Tensor( gnames_new  ) 
  gnames_new.requires_grad_( False )        
  rna_labels_new  = torch.Tensor(rna_labels_new).long()                                                    # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  rna_labels_new.requires_grad_( False )                                                                   # labels aren't allowed gradients


  # (iii) sanity check.  This situation can arise, and it will crash torch
  
  if  ( args.just_test!='True' ):
    
    if DEBUG>0:  
      print ( f"GENERATE:       INFO:  SANITY CHECK: number of unique classes represented in the cases                       = {MIKADO}{len(np.unique(rna_labels_new))}{RESET}"                                                            ) 
      print ( f"GENERATE:       INFO:  SANITY CHECK: classes in {CYAN}CLASS_NAMES{RESET}                                                  = {CYAN}{class_names}{RESET}"          ) 
    
    if len(np.unique(rna_labels_new)) != len(class_names):
      print ( f"{RED}GENERATE:       FATAL: different number of cancer types represented in the cases to be trained than in the configuration parameter {CYAN}CLASS_NAMES{RESET}{RESET}"                                          ) 
      print ( f"{RED}GENERATE:       FATAL:    number of unique classes represented in the cases    = {MIKADO}{len(np.unique(rna_labels_new))}{RESET}"                                                                            ) 
      print ( f"{RED}GENERATE:       FATAL:    classes in {CYAN}CLASS_NAMES{RESET}{RED}                               = {MIKADO}{len(class_names)}{RESET}{RED}, namely: {CYAN}{class_names}{RESET}"                               ) 
      print ( f"{BOLD}{RED}GENERATE:       FATAL: perhaps case division needs to be re-run to reflect changes made to the cases in the working directory (for example, changing to case {CYAN}-c {RESET}{BOLD}{RED}selection){RESET}"            )
      print ( f"{BOLD}{RED}GENERATE:       FATAL:    remedy: re-run the experiment with option {CYAN}-v {RESET}{BOLD}{RED} set to {CYAN}True{RESET}{BOLD}{RED} to have cases divided and flagged{RESET}"                          )
      print ( f"{BOLD}{RED}GENERATE:       FATAL:                  i.e. '{CYAN}./do_all.sh -d <cancer type code> -i <input type code> ... {CHARTREUSE}-v True{RESET}{BOLD}{RED}'{RESET}"                                          )
      print ( f"{BOLD}{RED}GENERATE:       FATAL: it's also possible that, by chance, no representatives of one of the smaller classes (subtypes) made it into the training set. Re-running with option {CYAN}-v {RESET}{BOLD}{RED} may remedy this (might need to try multiple times) {RESET}{CLEAR_LINE}",      flush=True  )                                        

      print ( f"{RED}GENERATE:       FATAL:    {RESET}"                                                                                                                                                                           )
      print ( f"{RED}GENERATE:       FATAL:    alternative remedy (1) include more cases to make it  more likely that examples of the missing class(es) will be represented{RESET}"                                               )
      print ( f"{RED}GENERATE:       FATAL:    alternative remedy (2) edit {CYAN}CLASS_NAMES{RESET}{RED} to only include the names of classes actually represented (but be careful: the order of {CYAN}CLASS_NAMES{RESET}{RED} has to be the same as the order of the class labels as represented in the master spreadsheet {CYAN}{args.cancer_type}_mapping_file_MASTER{RESET}{RED}, and 'gaps' are not permitted" )
      print ( f"{RED}GENERATE:       FATAL: halting now ...{RESET}", flush=True)
      sys.exit(0)    

  if DEBUG>2:
    print ( f"GENERATE:       INFO:  finished converting rna   data and labels     from numpy array to Torch tensor")
    print ( f"GENERATE:       INFO:    Torch size of genes_new       =  (~samples)                   {MIKADO}{genes_new.size()}{RESET}"      )
    print ( f"GENERATE:       INFO:    Torch size of gnames_new      =  (~samples)                   {MIKADO}{gnames_new.size()}{RESET}"     )
    print ( f"GENERATE:       INFO:    Torch size of rna_labels_new  =  (~samples)                   {MIKADO}{rna_labels_new.size()}{RESET}" )

  if DEBUG>88:
      print ( f"GENERATE:       INFO:    fnames_new                    =                               {MIKADO}{fnames_new}{RESET}"    )

  if DEBUG>2:        
      print ( f"GENERATE:       INFO:     rna_labels_new                 =                             \n{MIKADO}{rna_labels_new.numpy()}{RESET}"    )     
    

  # (6D) save numpy array for possible subsequent use by clustering functions

  if target=='rna_train':
    
    if DEBUG>0:  
      print( f"GENERATE:       INFO:    {COTTON_CANDY}now saving save numpy version of RNA-Seq vectors and labels arrays for possible subsequent use by clustering functions{RESET}{CLEAR_LINE}")
      
    fqn =  f"{args.base_dir}/logs/all_rna_seq_vectors_from_last_run_of_generate"
    np.save ( fqn, genes_new )
  
    fqn =  f"{args.base_dir}/logs/all_rna_seq_vector_labels_from_last_run_of_generate"
    np.save ( fqn, rna_labels_new )

  

  # (6E) save torch tensors as '.pth' file for subsequent loading by loader/dataset functions

  fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_{target}.pth"
  
  if DEBUG>2:  
    print( f"GENERATE:       INFO:  {WHITE}now saving as Torch dictionary (this takes a little time in the case of images){RESET}{CLEAR_LINE}")
    
  torch.save({
      'genes':      genes_new,
      'fnames':     fnames_new,
      'gnames':     gnames_new, 
      'rna_labels': rna_labels_new,           
  }, fqn )

    
  print( f"GENERATE:       INFO:  finished saving Torch dictionary to {MAGENTA}{fqn}{RESET}{CLEAR_LINE}" ) 



  return rna_files_processed, n_genes
  



  
#----------------------------------------------------------------------------------------------------------
def generate_image_dataset ( args, target, cases_required, highest_class_number, case_designation_flag, n_tiles_base, tiles_required, tile_size, top_up_factors  ):

  retrospective_class_counts = np.zeros( highest_class_number+1, dtype=np.int )
        
  cases_required = cases_required if cases_required>0 else 1                                               # zero cases will cause a crash 
  
  if DEBUG>0:
    print( f"{CLEAR_LINE}GENERATE:       INFO:     tiles_required                                                = {PINK}{tiles_required}{RESET}",        flush=True       ) 

  
  images_new      = np.ones ( ( tiles_required,  3, tile_size, tile_size ), dtype=np.uint8   )              
  fnames_new      = np.zeros( ( tiles_required                           ), dtype=np.int64   )             # np.int64 is equiv of torch.long
  img_labels_new  = np.zeros( ( tiles_required,                          ), dtype=np.int_    )             # img_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 

  if DEBUG>10:
    print( f"{CLEAR_LINE}GENERATE:       INFO:     making empty images_new.shape                                 = {PINK}{images_new.shape}{RESET}",             flush=True       ) 
    print( f"{CLEAR_LINE}GENERATE:       INFO:     making empty img_labels_new.shape                             = {PINK}{img_labels_new.shape}{RESET}",         flush=True       ) 
    print( f"{CLEAR_LINE}GENERATE:       INFO:     making empty fnames_new.shape                                 = {PINK}{fnames_new.shape}{RESET}",             flush=True       )


  global_tiles_processed  = 0
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
        if DEBUG>2:
          print ( f"{PALE_GREEN}GENERATE:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[130C has been tiled{RESET}{CLEAR_LINE}",  flush=True )
        if case_designation_flag=='ALL_ELIGIBLE_CASES':
          use_this_case_flag=True
        try:
          fqn = f"{dir_path}/{case_designation_flag}"        
          f = open( fqn, 'r' )
          use_this_case_flag=True
          if DEBUG>2:
            print ( f"{CLEAR_LINE}{GREEN}GENERATE:       INFO:   case \r\033[55C'{COTTON_CANDY}{dir_path}{RESET}{GREEN}' \r\033[130C is     a case flagged as '{CYAN}{case_designation_flag}{RESET}{GREEN}' - - including{RESET}{CLEAR_LINE}",  flush=True )
        except Exception:
          if DEBUG>4:
            print ( f"{RED}GENERATE:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED} \r\033[130C is not a case flagged as '{CYAN}{case_designation_flag}{RESET}{RED}' - - skipping{RESET}{CLEAR_LINE}",  flush=True )
      except Exception:
        if DEBUG>4:
          print ( f"{PALE_RED}GENERATE:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C has not been tiled{RESET}{CLEAR_LINE}",  flush=True )

      try:                                                                                                 # every tile has an associated label - the same label for every tile image in the directory
        label_file = os.path.join(dir_path, args.class_numpy_file_name)
        label      = np.load( label_file )
        subtype    = label[0] 
        if subtype>highest_class_number:
          use_this_case_flag=False
          if DEBUG>2:
            print ( f"{ORANGE}GENERATE:       INFO: label is greater than '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{ORANGE}' - - skipping this example (label = {MIKADO}{subtype}{RESET}{ORANGE}){RESET}"      )
          pass
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL: when processing: '{label_file}'{RESET}", flush=True)        
        print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)
        sys.exit(0)
    
    
    
    
    if ( use_this_case_flag==True ):
      
      # if user has requested 'MAKE_BALANCED', then adjust the number of tiles to be including according to the subtype using the 'top_up_factors' array
      # the tiling process will have ensured that at least this many tiles is available
      
      if ( (args.make_balanced=='level_up') | (args.make_balanced=='level_down')  ):
 
        if DEBUG>2:
          print ( f"\r{CLEAR_LINE}{BOLD}{CARRIBEAN_GREEN}GENERATE:       INFO:   base value of n_tiles       = {CYAN}{n_tiles_base}{RESET}"            )
          np.set_printoptions(formatter={'float': lambda x: "{:6.2f}".format(x)})
          print ( f"\r{CLEAR_LINE}{BOLD}{CARRIBEAN_GREEN}GENERATE:       INFO:   tile top_up_factors         = {CYAN}{top_up_factors}{RESET}"                  )
          print ( f"\r{CLEAR_LINE}{BOLD}{CARRIBEAN_GREEN}GENERATE:       INFO:   applicable top up factor    = {CYAN}{top_up_factors[subtype]:<4.2f}{RESET}"   )
            
        if top_up_factors[subtype]==1.:                                                                      # no need to adjust n_tiles for the subtype which has the largest number of images
          n_tiles = n_tiles_base
        else:
          tiles_needed_by_subtype = np.around((top_up_factors*n_tiles_base), 0).astype(int)
          tiles_needed_by_subtype = np.array( [ el if el!=0 else 1 for el in tiles_needed_by_subtype ] )  
          n_tiles = tiles_needed_by_subtype[subtype]
  
        if DEBUG>2:
          print ( f"\r{CLEAR_LINE}{BOLD}{CARRIBEAN_GREEN}GENERATE:       INFO:   new value of n_tiles        = {CYAN}{n_tiles}{RESET}"       )
            
      else:
        n_tiles = n_tiles_base




      if DEBUG>2:
        print( f"{CLEAR_LINE}{CARRIBEAN_GREEN}GENERATE:       INFO:   now processing case:           '{CAMEL}{dir_path}{RESET}'{CLEAR_LINE}" )
        
      # (2) Set up symlink

      for f in sorted( files ):  

        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
          
          fqsn                      = f"{dir_path}/entire_patch.npy"
          parent_dir                = os.path.split(os.path.dirname(fqsn))[1]
          no_special_chars_version  = re.sub('[^A-Za-z0-9]+', '', parent_dir).lstrip()
          final_chars               = no_special_chars_version[-6:]
          int_version               = int( final_chars, 16)
              
          if DEBUG>5:
            print (f"GENERATE:       INFO:  fully qualified file name of slide = '{MAGENTA}{fqsn}{RESET}'"                     )
            print (f"GENERATE:       INFO:                            dir_path = '{MAGENTA}{dir_path}{RESET}'"                 )
            print (f"GENERATE:       INFO:                          parent_dir = '{MAGENTA}{parent_dir}{RESET}'"               )
            print (f"GENERATE:       INFO:            no_special_chars_version = '{MAGENTA}{no_special_chars_version}{RESET}'" )
            print (f"GENERATE:       INFO:                         final_chars = '{MAGENTA}{final_chars}{RESET}'"              )
            print (f"GENERATE:       INFO:                         hex_version = '{MAGENTA}{int_version}{RESET}'"              )
    
    
          svs_file_link_id   = int_version
          svs_file_link_name = f"{svs_file_link_id:d}"
    
          fqln = f"{args.data_dir}/{svs_file_link_name}.fqln"                                                  # name for the link
          try:
            os.symlink( fqsn, fqln)                                                                            # make the link
          except Exception as e:
            if DEBUG>2:
              print ( f"{ORANGE}GENERATE:       NOTE:  Link already exists{RESET}" )
            else:
              pass
    
          if DEBUG>2:
            print( f"GENERATE:       INFO:                                fqln = '{MAGENTA}{fqln}{RESET}{CLEAR_LINE}'"                     )   
          if DEBUG>90:
            print( f"GENERATE:       INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}"          )
            print( f"GENERATE:       INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'"       )


      # (3) set up the array for each png entry in this directory
      
      tile_extension  = "png"
    
      for f in sorted( files ):                                                                                # examine every file in the current directory
               
        if DEBUG>999:
          print( f"GENERATE:       INFO:               files                  = {MAGENTA}{files}{RESET}"      )
      
        image_file    = os.path.join(dir_path, f)
    
        if DEBUG>2:  
          print( f"GENERATE:       INFO:     image_file    = {CYAN}{image_file}{RESET}", flush=True   )
          print( f"GENERATE:       INFO:     label_file    = {CYAN}{label_file}{RESET}",   flush=True   )
    
        
        if ( f.endswith('.' + tile_extension ) & (not ( 'mask' in f ) ) & (not ( 'ized' in f ) )   ):      # because there may be other png files in each image folder besides the tile image files
    
          try:
            img = cv2.imread( image_file )
            if DEBUG>2:
              print ( f"GENERATE:       INFO:     image_file    =  \r\033[55C'{AMETHYST}{image_file}{RESET}'"    )
          except Exception as e:
            print ( f"{RED}GENERATE:             FATAL: when processing: '{image_file}'{RESET}", flush=True)    
            print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
            print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)
            sys.exit(0)    
    
    
          try:
            images_new [global_tiles_processed,:] =  np.moveaxis(img, -1,0)                                # add it to the images array
          except Exception as e:
            print ( f"{RED}GENERATE:             FATAL:  [1320] reported error was: '{e}'{RESET}", flush=True )
            print ( f"{RED}GENERATE:                      Explanation: The dimensions of the array reserved for tiles is  {MIKADO}{images_new [global_tiles_processed].shape}{RESET}{RED}; whereas the tile dimensions are: {MIKADO}{np.moveaxis(img, -1,0).shape}{RESET}", flush=True )                 
            print ( f"{RED}GENERATE:                      {RED}Did you change the tile size without regenerating the tiles? {RESET}", flush=True )
            print ( f"{RED}GENERATE:                      {RED}Either run'{CYAN}./do_all.sh -d <cancer type code> -i image{RESET}{RED}' to generate {MIKADO}{images_new [global_tiles_processed].shape[1]}x{images_new [global_tiles_processed].shape[1]}{RESET}{RED} tiles, or else change '{CYAN}TILE_SIZE{RESET}{RED}' to {MIKADO}{np.moveaxis(img, -1,0).shape[1]}{RESET}", flush=True )                 
            print ( f"{RED}GENERATE:                      {RED}Halting now{RESET}", flush=True )                 
            sys.exit(0)
    
          try:                                                                                             # every tile has an associated label - the same label for every tile image in the directory
            label = np.load( label_file )
            if DEBUG>99:
              print ( f"GENERATE:       INFO:     label.shape   =  {MIKADO}{label.shape}{RESET}"   )
            if DEBUG>99:         
              print ( f"GENERATE:       INFO:     label value   =  {MIKADO}{label[0]}{RESET}"      )
            if DEBUG>2:
              print ( f"{AMETHYST}{label[0]}", end=', ', flush=True )
          except Exception as e:
            print ( f"{RED}GENERATE:             FATAL: when processing: '{label_file}'{RESET}", flush=True)        
            print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
            print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)
            sys.exit(0)
                                    
          img_labels_new[global_tiles_processed] = label[0]                                                # add it to the labels array
          retrospective_class_counts[label[0]]+=1                                                          # keep track of the number of examples of each class 
          
          #img_labels_new[global_tiles_processed] =  random.randint(0,5)                                   # swap truth labels to random numbers for testing purposes
    
          if DEBUG>77:  
            print( f"GENERATE:       INFO:     label                  = {MIKADO}{label[0]:<8d}{RESET}", flush=True   )
            
    
          fnames_new [global_tiles_processed]  =  svs_file_link_id                                         # link to filename of the slide from which this tile was extracted - see above
    
          if DEBUG>88:
              print( f"GENERATE:       INFO: symlink for tile (fnames_new [{BLEU}{global_tiles_processed:3d}{RESET}]) = {BLEU}{fnames_new [global_tiles_processed]}{RESET}" )
          
    
          if DEBUG>100:
            print ( "=" *180)
            print ( "GENERATE:       INFO:          tile {:} for this image:".format( global_tiles_processed+1))
            print ( "GENERATE:       INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
            print ( "GENERATE:       INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
          if DEBUG>100:
            print ( f"GENERATE:       INFO:                value = \n{images_new[global_tiles_processed,0,0,:]}")
    
          the_class=img_labels_new[global_tiles_processed]
          if the_class>3000:
              print ( f"{RED}GENERATE:       FATAL: ludicrously large class value detected (class={MIKADO}{the_class}{RESET}{RED}) for tile '{MAGENTA}{image_file}{RESET}" )
              print ( f"{RED}GENERATE:       FATAL: hanting now [1718]{RESET}" )
              sys.exit(0)
              
    
          if DEBUG>66:
            print ( "GENERATE:       INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
            print ( "GENERATE:       INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
            print ( "GENERATE:       INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
           
          global_tiles_processed+=1
          
          tiles_processed+=1
          if tiles_processed==n_tiles:
            break
          
        else:
          if DEBUG>44:
            print( f"GENERATE:       INFO:          other file = {MIKADO}{image_file}{RESET}".format(  ) )

        
      if DEBUG>2:
        print( f"GENERATE:       INFO:   tiles processed in directory: \r\033[55C'{MAGENTA}{dir_path}{RESET}'  \r\033[130C= {MIKADO}{tiles_processed:<8d}{RESET}",        flush=True       )   
        
      if args.just_test!='True':
        if tiles_processed!=n_tiles:
          print( f"{CLEAR_LINE}{BOLD}{ORANGE}GENERATE:       INFO:   for directory: \r\033[55C'{MAGENTA}{dir_path}{RESET}{BOLD}{ORANGE}' \r\033[134Ctiles processed = {MIKADO}{tiles_processed:<8d}{BOLD}{ORANGE} instead of expected {MIKADO}{n_tiles}{RESET}{BOLD}{ORANGE}                   <<<<<<<<<<<< anomoly {RESET}", flush=True  )       

      directories_processed+=1
      if DEBUG>2:
        print( f"{CAMEL}GENERATE:       INFO:   {CYAN}cases_required{RESET}{CAMEL}        = {MIKADO}{cases_required:<4d}{RESET})",          flush=True        )  
        print( f"{CAMEL}GENERATE:       INFO:   {CYAN}directories_processed{RESET}{CAMEL} = {MIKADO}{directories_processed:<4d}{RESET}",    flush=True        )   
      if directories_processed>=cases_required:
        sufficient_cases_found=True
        if DEBUG>0:
          print( f"{CAMEL}GENERATE:       INFO:   sufficient directories were found ({CYAN}cases_required{RESET}{CAMEL} = {MIKADO}{cases_required:<4d}{RESET})",  flush=True        )  
        break

  if sufficient_cases_found!=True:
    
    if args.just_test=='True':
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:   (test mode) the number of cases found and processed ({MIKADO}{directories_processed}{RESET}{BOLD_ORANGE}) is less than the number required ({MIKADO}{cases_required}{RESET}{BOLD_ORANGE})",  flush=True        ) 
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:   (test mode)   possible explanation: if you set '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{BOLD_ORANGE}' to a number less than the number of classes (subtypes) actually present in the dataset, none of the '{CYAN}{args.cases}{RESET}{BOLD_ORANGE}' cases belonging to the classes (subtypes) you removed will be available to be used",  flush=True        )
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:   (test mode)   not halting, but this will likely cause problems",  flush=True      )
      time.sleep(4)
    else:
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:   the number of cases found and processed ({MIKADO}{directories_processed}{RESET}{BOLD_ORANGE}) is less than the number required ({MIKADO}{cases_required}{RESET}{BOLD_ORANGE})",  flush=True        ) 
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:     one possible explanation: if you set '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{BOLD_ORANGE}' to a number less than the number of classes actually present in the dataset, none of the '{CYAN}{args.cases}{RESET}{BOLD_ORANGE}' cases belonging to the classes you removed will be available to be used",  flush=True        )
      print( f"{CLEAR_LINE}{BOLD_ORANGE}GENERATE:       WARNG:     not halting, but this might cause problems",  flush=True      )
  

  
  if DEBUG>0:
    print( f"{CLEAR_LINE}{ASPARAGUS}GENERATE:       INFO:   directories_processed = {MIKADO}{directories_processed:<4d}{RESET}",  flush=True        )   

  if DEBUG>0:
    print( f"{BOLD_PEWTER}GENERATE:       INFO:     images_new.shape               = {MIKADO}{images_new.shape}{RESET}",    flush=True       ) 
    print( f"GENERATE:       INFO:     fnames_new.shape               = {MIKADO}{fnames_new.shape}{RESET}",             flush=True       )
    print( f"GENERATE:       INFO:     img_labels_new.shape           = {MIKADO}{img_labels_new.shape}{RESET}",         flush=True       )
  if DEBUG>2:
    print( f"GENERATE:       INFO:     img_labels_new                 = \n{MIKADO}{img_labels_new}{RESET}",             flush=True       )


  if DEBUG>0:
    print( f"{CLEAR_LINE}GENERATE:       INFO:    tiles drawn from each of the {MIKADO}{highest_class_number+1}{RESET} cancer sub/types                                 = {MIKADO}{retrospective_class_counts}{RESET}",   flush=True       )
    print( f"{CLEAR_LINE}GENERATE:       INFO:    total tiles generated (actual)                                                = {MIKADO}{np.sum(retrospective_class_counts)}{RESET}",                                   flush=True       )
    print( f"{CLEAR_LINE}GENERATE:       INFO:    for reference, space was allocated in advance for this many tiles               {MIKADO}{tiles_required}{RESET}",                                                       flush=True       )




  # save numpy array for possible subsequent use by clustering functions

  if target=='image_train':
    
    if DEBUG>0:  
      print( f"GENERATE:       INFO:    {COTTON_CANDY}now saving save numpy version of image and labels arrays for possible subsequent use by clustering functions{RESET}{CLEAR_LINE}")
      
    fqn =  f"{args.base_dir}/logs/all_images_from_last_run_of_generate"
    np.save ( fqn, images_new )
  
    fqn =  f"{args.base_dir}/logs/all_image_labels__from_last_run_of_generate"
    np.save ( fqn, img_labels_new )

    if DEBUG>0:
      time.sleep(0.5)
      print( f"\r\033[1AGENERATE:       INFO:    {BRIGHT_GREEN}numpy version of image and labels arrays saved for possible subsequent use by clustering functions{RESET}{CLEAR_LINE}\033[1B")


  # trim, then convert everything into Torch style tensors

  images_new      = images_new     [0:global_tiles_processed]                                            # trim off all the unused positions in image_new
  img_labels_new  = img_labels_new [0:global_tiles_processed]                                            # ditto
  fnames_new      = fnames_new     [0:global_tiles_processed]                                            # ditto
  
  if args.dataset =='skin':                                                                                     # CIFAR10 is a special case. Pytorch has methods to retrieve cifar and some other benchmakring databases, and stores them in a format that is ready for immediate loading, hence earlier steps like tiling, and also the generation steps that have to be applied to GDC datasets can be skipped
  
    if ( tile_size<299) & ( (any( 'INCEPT'in item for item in args.nn_type_img ) ) ):

      print( f"{BOLD_ORANGE}GENERATE:       WARN: tile_size = {MIKADO}{tile_size}{BOLD_ORANGE}. However, for '{CYAN}NN_TYPE_IMG{BOLD_ORANGE}={MIKADO}INCEPT4{BOLD_ORANGE}', the tile size must be at least {MIKADO}299x299{RESET}" )
      print( f"{BOLD_ORANGE}GENERATE:       WARN: upsizing the {CYAN}tile_size{BOLD_ORANGE} of all images to {MIKADO}299x299{RESET}{RESET}" )
      
      items = images_new.shape[0]
      images_new_upsized = np.zeros( ( items, 299, 299, 3 ), dtype=np.uint8  )
      
      for i in range ( 0, images_new.shape[0] ):      

        if i%100:
          print ( f"\rGENERATE:       INFO: {i+1} of {items} images have been upsized", end='', flush=True)
        
        item = images_new[i]
        
        xform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299))
            #transforms.RandomRotation((0, 360)),
            #transforms.RandomCrop(cfg.IMG_SIZE),
            #transforms.RandomHorizontalFlip(),
            # ~ transforms.ToTensor()
        ])
        
        images_new_upsized[i]  = xform(item) 
  
      print( "" )
      images_new = images_new_upsized
      tile_size = args.tile_size = 299

  images_new  = images_new.swapaxes(1,3)                                                                   # it's stored as XXX,HH,WW,3 whereas we need XXX,3,HH,WW so we swap second and last axes around
  
  images_new      = torch.Tensor ( images_new )
  fnames_new      = torch.Tensor ( fnames_new ).long()
  fnames_new.requires_grad_      ( False )
  img_labels_new  = torch.Tensor ( img_labels_new ).long()                                                 # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  img_labels_new.requires_grad_  ( False )

  if DEBUG>1:
    print( "\nGENERATE:       INFO:   finished converting image data and labels from numpy array to Torch tensor")

  if DEBUG>0:
    print( f"{BOLD_ASPARAGUS}GENERATE:       INFO:     images_new.size()               = {MIKADO}{images_new.size()}{RESET}",    flush=True       ) 

  if DEBUG>2:
    print ( f"GENERATE:       INFO:     img_labels_new                =                               \n{MIKADO}{img_labels_new}{RESET}{CLEAR_LINE}"    )  



  # save torch tensors as '.pth' file for subsequent loading by dataset function

  fqn =  f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/dataset_{target}.pth"
  
  if DEBUG>0:  
    print( f"GENERATE:       INFO:    {PINK}now saving as Torch dictionary (this can take some time in the case of images){RESET}{CLEAR_LINE}")
  
  torch.save({
      'images':     images_new,
      'fnames':     fnames_new,
      'img_labels': img_labels_new,
  }, fqn )

    
  print( f"GENERATE:       INFO:    finished saving Torch dictionary to {MAGENTA}{fqn}{RESET}{CLEAR_LINE}" )
  

  return directories_processed, global_tiles_processed
  


#----------------------------------------------------------------------------------------------------------
# HELPER FUNCTION
#----------------------------------------------------------------------------------------------------------

def check_mapping_file ( args, case ):

  if args.mapping_file_name=='none':
    return True
  else:                                                                                                    # a custom mapping file is active
  # then look inside the custom mapping file to see if this case exists
    exists = True
    return ( exists )

