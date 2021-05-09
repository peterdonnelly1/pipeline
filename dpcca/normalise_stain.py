"""
open each rna file in data_dir; extract 'rna_exp_column' (column 1); convert to numpy array; save numpy array as a new file with the same name but .npy extension 

"""

import os
import sys
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

import glob
import tensorflow as tf

print (tf.Session())

from Run_StainSep  import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm

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
  

  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
  # config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
  config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
  
  
  #Parameters
  nstains=2    #number of stains
  lamb=0.01     #default value sparsity regularization parameter
  # lamb=0 equivalent to NMF

  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: will look recursively under:      {MAGENTA}{data_dir}{RESET} for slide files (files ending with either 'svs' or 'SVS'",  flush=True ) 

  slide_file_found = 0
  
  for dir_path, __, files in os.walk( data_dir):
    
    for f in sorted(files):
    
      current_file    = os.path.join( dir_path, f)
  
      if (DEBUG>0):
        print ( f"NORMALISE_STAIN:        INFO: (current_file)                    {DULL_BLUE}{current_file}{RESET}",  flush=True )

  
      if ( f.endswith( 'svs' ) )  |  ( f.endswith( 'SVS' )  ):
   
        slide_file_found += 1

        if slide_file_found==22:
            
          if (DEBUG>0): 
            print ( f"NORMALISE_STAIN:        INFO: (match !)                         {BRIGHT_GREEN}{current_file}{RESET}    slide files found so far = {ARYLIDE}{slide_file_found}{RESET}",  flush=True )
  
          if (DEBUG>0):
            print ( f"NORMALISE_STAIN:        INFO: about to stain separate           {GOLD}{current_file}{RESET}",  flush=True )
            
          run_stainsep( current_file, nstains,lamb )
          # ~ Wi,Hi,Hiv,sepstains = run_stainsep( current_file, nstains,lamb )
  
          source_filename = current_file
          
          if (DEBUG>0):
            print ( f"NORMALISE_STAIN:        INFO: successfully stain separated      {GREEN}{current_file}{RESET}",  flush=True )
            print ( f"NORMALISE_STAIN:        INFO:                          Wi   =   {MIKADO}{Wi}{RESET}",           flush=True )
            print ( f"NORMALISE_STAIN:        INFO:                          Wi   =   {MIKADO}{Wi}{RESET}",           flush=True )
            print ( f"NORMALISE_STAIN:        INFO:                          Hi   =   {MIKADO}{Hi}{RESET}",           flush=True )
            print ( f"NORMALISE_STAIN:        INFO:                   sepstains   =   {MIKADO}{sepstains}{RESET}",    flush=True )

                  
        if slide_file_found==3:

          nstains               = 2                                                                        # number of stains
          lamb                  = 0.01                                                                     # default value sparsity regularization parameter
          level                 = 0
          background_correction = True
          
          run_colornorm( current_file, current_file, nstains, lamb, dir_path, level, background_correction, config=config )
          
          break
        

      
#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                type=str, default="/home/peter/git/pipeline/dataset")
  
  args, _ = p.parse_known_args()

  main(args)
      
