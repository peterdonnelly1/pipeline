"""
based on ...................... (insert reference & URL to author) 

The reference image ('source') is currently hardwired, and was obtained as follows:

1 From dataset, identify an notionally typical svs image: i.e. typical of the STAD dataset; with few processing artefacts (such as folds and tears); topologically contiguous (no significant voids) and not too much background:

    this one >> dataset/e42f45c6-fd00-44cf-a210-c8803da326a1_1/TCGA-CD-5798-01A-01-BS1.e1b74bd0-cf94-4356-8a6c-87ddb2d31c8f.svs

2 Use libvips to convert to tif format, and save to the top level of dataset:
    
    vips extract_band TCGA-CD-5798-01A-01-BS1.e1b74bd0-cf94-4356-8a6c-87ddb2d31c8f.svs Z.tif[pyramid,tile,compression=jpeg,tile-width=256,tile-height=256] 0 --n 1

    vips tiffsave TCGA-CD-5798-01A-01-BS1.e1b74bd0-cf94-4356-8a6c-87ddb2d31c8f.svs x.tif --pyramid --tile --tile-width=256 --tile-height=256

    dataset/TCGA-CD-5798-01A-01-BS1.e1b74bd0-cf94-4356-8a6c-87ddb2d31c8f.svs

3 Extract a large, representative, background free portion of the tif file with gimp to yield:

    dataset/TCGA-CD-5798-01A-01-BS1.e1b74bd0-cf94-4356-8a6c-87ddb2d31c8f_SPCN_REFERENCES.tif


"""

import os
import sys
import glob
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}                                         # This has to be before the import
import tensorflow as tf
print (tf.Session())

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

from Run_StainSep  import run_stainsep
from Run_ColorNorm import run_batch_colornorm

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
CHARTREUSE='\033[38;2;223;255;0m'
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

DEBUG   = 1

FAIL    = 0
SUCCESS = 1


#====================================================================================================================================================
def main(args):

  data_dir                 = args.data_dir
  
  reference_file = f"{data_dir}/TCGA-IP-7968-11A-01-TS1.aa84dfd6-6660-4488-b7d6-7652445a6f35.svs"

  gpu_options=tf.GPUOptions( per_process_gpu_memory_fraction=1 )
  # config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
  config = tf.ConfigProto( log_device_placement=False, gpu_options=gpu_options )


  nstains               = 2                                                                                # number of stains
  lamb                  = 0.01                                                                             # default value sparsity regularization parameter. lamb=0 equivalent to NMF
  level                 = 0
  background_correction = True

  is_reference_file = 0
  
  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: about to characterise designated reference file:   {CARRIBEAN_GREEN}{reference_file}{RESET}",  flush=True ) 
  target_i0,  Wi_target, Htarget_Rmax, normalisation_factor =  run_batch_colornorm  ( is_reference_file,  reference_file, reference_file,  nstains,  lamb,  data_dir, level, background_correction, 0,0,0,0,     config  )
  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: reference file characterised                       {CARRIBEAN_GREEN}{reference_file}{RESET}",  flush=True ) 


  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: will look recursively under:                       {MAGENTA}{data_dir}{RESET} for slide files (files ending with either 'svs' or 'SVS')\n",  flush=True ) 

  slide_file_found  = 0
  is_reference_file = 1
  
  already_processed_this_slide=False
    
  for dir_path, __, files in os.walk( data_dir):

    if not (dir_path==args.data_dir):                                                                      # the top level directory (dataset) has to be skipped because it contains the reference svs file    

      if already_processed_this_slide==True:

        if (DEBUG>0):
          print ( f"{ORANGE}NORMALISE_STAIN:        INFO: normalised version exists ... skipping     ({DULL_BLUE}{current_file}{RESET})",    flush=True )
      
        already_processed_this_slide=False
          
      else:
      
        for f in sorted(files):
        
          current_file = f"{dir_path}/{f}"
      
          if (DEBUG>2):
            print ( f"NORMALISE_STAIN:        INFO: (current_file)                                     {DULL_BLUE}{current_file}{RESET}",    flush=True )
            print ( f"NORMALISE_STAIN:        INFO: (reference_file)                                   {DULL_BLUE}{reference_file}{RESET}",  flush=True )
            # ~ print ( f"NORMALISE_STAIN:        INFO: ( reference_file[-40:])                        {DULL_BLUE}{ reference_file[-40:]}{RESET}",  flush=True )
  
          if ( f.endswith( 'spcn' )  ):                                                                    # this folder has already been handled, so moveon to the next folder
            if (DEBUG>0):
              print ( f"NORMALISE_STAIN:        INFO: a file with extension {CYAN}spcn{RESET} exists in this folder, so will move on to the next folder",  flush=True )
            already_processed_this_slide=True 
            break
      
          if ( f.endswith( 'svs' ) )  |  ( f.endswith( 'SVS' )  ):
       
            slide_file_found += 1
    
            # ~ if slide_file_found==1:
                
            if (DEBUG>0): 
              print ( f"NORMALISE_STAIN:        INFO: found an svs file                                  {BRIGHT_GREEN}{current_file}{RESET}    slide files found so far = {ARYLIDE}{slide_file_found}{RESET}",  flush=True )
                
              # ~ run_stainsep ( current_file, nstains, lamb  )
              # Wi,Hi,Hiv,sepstains = run_stainsep( current_file, nstains,lamb )
      
              # ~ if (DEBUG>0):
                # ~ print ( f"NORMALISE_STAIN:        INFO: successfully stain separated      {GREEN}{current_file}{RESET}",  flush=True )
                # ~ print ( f"NORMALISE_STAIN:        INFO:                          Wi   =   {MIKADO}{Wi}{RESET}",           flush=True )
                # ~ print ( f"NORMALISE_STAIN:        INFO:                          Wi   =   {MIKADO}{Wi}{RESET}",           flush=True )
                # ~ print ( f"NORMALISE_STAIN:        INFO:                          Hi   =   {MIKADO}{Hi}{RESET}",           flush=True )
                # ~ print ( f"NORMALISE_STAIN:        INFO:                   sepstains   =   {MIKADO}{sepstains}{RESET}",    flush=True )
    
    
            if (DEBUG>0):
              print ( f"NORMALISE_STAIN:        INFO: about to colour normalise:                         {GOLD}{current_file}{RESET}",  flush=True )          
              print ( f"NORMALISE_STAIN:        INFO: dir_path                                           {GOLD}{dir_path}{RESET}",      flush=True )          
              _,  _, _, _   =  run_batch_colornorm  ( is_reference_file, current_file, reference_file,  nstains,  lamb,  dir_path, level, background_correction, target_i0,  Wi_target, Htarget_Rmax, normalisation_factor, config  )
            if (DEBUG>0):
              print ( f"NORMALISE_STAIN:        INFO: colour normalisation complete",  flush=True )
            
      
#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                type=str, default="/home/peter/git/pipeline/dataset")
  
  args, _ = p.parse_known_args()

  main(args)
      
