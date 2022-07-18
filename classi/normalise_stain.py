# all code used for spcn stain normalisation is due to:
#     D. Anand, G. Ramakrishnan, A. Sethi
#     and is an implementation of spcn described in their paper:
#        "Fast GPU-enabled color normalization for digital pathology"
#        International Conference on Systems, Signals and Image Processing, Osijek, Croatia (2019), pp. 219-224
#
#
# the code I used (trivially modified by me) may be found here:
#     https://github.com/goutham7r/spcn
#
#
# their spcn implementation is in turn an optimised, GPU based version of the original spcn algorithm which was created by:
#     Vahadane, A. et al.
#     as described in their paper: 
#         "Structure-preserving color normalization and sparse stain separation for histological images" 
#         IEEE Trans. Med. Imaging. 35, 1962–1971 (2016).
#
#
#
#
#  Notes:
#   spcn stain normalisation takes place outside of the CLASSI framework
#   it is run prior to using CLASSI, and creates a new, stain normalised version of each SVS file it finds in the working data directory and places it in the same directory
#   when the spcn option is selected in CLASSI, (-0 spcn), these normalised files are used rather than the SVS files    
#   
#   further:
#   1`characterising the reference file typically takes a long time - perhaps half an hour
#   2`stain normalisation of svs files, which are typically very large, likewise can take a long time - e.g 10-30 minutes per image
#   2 the program performing spcn stain normalisation uses tensorflow rather than pytorch
#   3 since it uses some of the same libraries as CLASSI, but at different version levels, it should be run in a different virtual environment to CLASSI (I use conda)
#   4 here are the (original) dependencies:
#        python              3.6.13
#        tensorflow          1.15.0
#        numpy               1.19.5
#        pyvips              2.1.8
#        openslide           3.4.1
#        openslide-python    1.1.2
#        pillow              8.1.2
#        spams               2.6.1
#        scikit-learn        0.23.2

import os
import sys
import ast
import psutil
import pickle
import numpy as np
import argparse
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


'''
Disable Tensorflow logging level.  This has to be before the import

  0 = all messages are logged (default behavior)
  1 = INFO messages are not printed
  2 = INFO and WARNING messages are not printed
  3 = INFO, WARNING, and ERROR messages are not printed
  
'''

import tensorflow as tf
print (tf.Session())

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

from Run_StainSep  import run_stainsep
from Run_ColorNorm import run_batch_colornorm

from constants  import *

DEBUG   = 1


#====================================================================================================================================================
def main(args):


  num_cpus  = multiprocessing.cpu_count()
  pid       = os.getpid()
  process   = psutil.Process(pid)
  memoryUse = process.memory_info()[0]/2.**30  # memory use in GB...I think
  affinity  = os.sched_getaffinity(pid)
   
            
  if DEBUG>0:
    print( f'{SAVE_CURSOR}{CLEAR_LINE}{RESET}  status {MIKADO}{process.status()}{RESET}  affinity {MIKADO}{affinity}{RESET}  pid {MIKADO}{pid:>6d}{RESET}   memory use: {MIKADO}{100*memoryUse:3.1f}{RESET}%   {CLEAR_LINE}{RESTORE_CURSOR}')
  
  # added this in Jun 2022 because my AMD computer started using only one of the 32 available cores
  # apparently others have had this issue:see e.g. https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
  x = {i for i in range(num_cpus)}
  os.sched_setaffinity( pid, x)

  force_reference_file_characterisation = args.force_reference_file_characterisation
  dataset                               = args.dataset
  data_source                           = f"{args.data_source}/{dataset}"
  reference_file                        = f"{data_source}/{args.reference_file}"


  if (DEBUG>0):
    print ( f"{BOLD}{MAGENTA}NORMALISE_STAIN:        IMPORTANT: if you have changed the svs reference file, you must manually delete all existing files with extension '{BOLD}{CYAN}.spcn{RESET}{BOLD}{MAGENTA}' in the applicable source data directory, namely: {BOLD}{CYAN}{data_source}{RESET}",  flush=True )
         

  if DEBUG>0:
    print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO: data_source    = {CYAN}{data_source}{RESET}{ORANGE}{RESET}" )    
    print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO: reference_file = {CYAN}{reference_file}{RESET}{ORANGE}{RESET}" )    

  gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=1 )
  # config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
  config      = tf.ConfigProto( log_device_placement=False, gpu_options=gpu_options )
  


  nstains               = 2                                                                                # number of stains
  lamb                  = 0.01                                                                             # default value sparsity regularization parameter. lamb=0 equivalent to NMF
  level                 = 0
  background_correction = True



  # (1) MAYBE CHARACTERISE THE REFERENCE FILE (WHICH IS JUST AN SVS FILE CHOSEN BY THE USER TO BE SUITABLE REPRESENTATIVE OF ALL THE SLIDES

  if os.path.isfile(reference_file)!=True:
    print ( f"{BOLD}{RED}NORMALISE_STAIN:        FATAL:  the image reference file provided ('{CYAN}{reference_file}{RESET}{RED}') does not exist.{RESET}" )
    print ( f"{BOLD}{RED}NORMALISE_STAIN:        FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)

  ref_file_characterisation_fname = f"{reference_file}.spcn_characterisation_details.pickle"

  if args.force_reference_file_characterisation == 'False':

    if os.path.exists (ref_file_characterisation_fname ):                                                  # user's selected reference file has previously been characterised, so we use existing characterisation, (saves about 30 minutes) 
      
      if DEBUG>0:
        print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO:  image characterisation details for this svs file exists from an earlier run; in file: {CYAN}{ref_file_characterisation_fname}{RESET}" )    
        print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO:  these will be loaded and used{RESET}" ) 
        
      with open( f"{ref_file_characterisation_fname}", "rb") as file:
        ref_file_characterisation = pickle.load(file)    

      target_i0            = ref_file_characterisation["target_i0"]
      Wi_target            = ref_file_characterisation["Wi_target"]
      Htarget_Rmax         = ref_file_characterisation["Htarget_Rmax"]
      normalisation_factor = ref_file_characterisation["normalisation_factor"]

      if DEBUG>0:
        print ( f"{ORANGE}NORMALISE_STAIN:        INFO:      target_i0            = {MIKADO}{target_i0}{RESET}"            ) 
        print ( f"{ORANGE}NORMALISE_STAIN:        INFO:      Htarget_Rmax         = {MIKADO}{Htarget_Rmax}{RESET}"         ) 
        print ( f"{ORANGE}NORMALISE_STAIN:        INFO:      normalisation_factor = {MIKADO}{normalisation_factor}{RESET}" ) 
        print ( f"{ORANGE}NORMALISE_STAIN:        INFO:      Wi_target            = \n{MIKADO}{Wi_target}{RESET}"          ) 

  else:                                                                                                    # user's selected reference file has not previously been characterised, so we characterise it now

    if DEBUG>0:
      if force_reference_file_characterisation != 'False':
        print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO: NOTE !!! config option {BOLD_CYAN}force_reference_file_characterisation{RESET}{BOLD_ORANGE} = {RESET}{MIKADO}{force_reference_file_characterisation}{RESET}{BOLD_ORANGE}. The reference file will be (re)characterised."    )    
      else:
        print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO: the chosen reference file has not been previously characterised{RESET}"    )    

      print ( f"NORMALISE_STAIN:        INFO: about to characterise designated reference file:   {CYAN}{reference_file}{RESET}"        ) 
    

    is_reference_file = 0                                                                                  # 0=reference file; 1=any other svs file
    target_i0,  Wi_target, Htarget_Rmax, normalisation_factor =  run_batch_colornorm  ( is_reference_file,  reference_file, reference_file,  nstains,  lamb,  data_source, level, background_correction, 0,0,0,0,  config  )

    if (DEBUG>0):
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO: reference file has now been characterised                       {MAGENTA}{reference_file}{RESET}"  ) 
  
    if DEBUG>0:
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO:  about to save image characterisation details for possible future use, to file: {CYAN}{ref_file_characterisation_fname}{RESET}" )    
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO:      target_i0            = {MIKADO}{target_i0}{RESET}"            ) 
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO:      Htarget_Rmax         = {MIKADO}{Htarget_Rmax}{RESET}"         ) 
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO:      normalisation_factor = {MIKADO}{normalisation_factor}{RESET}" ) 
      print ( f"{CAMEL}NORMALISE_STAIN:        INFO:      Wi_target            = \n{MIKADO}{Wi_target}{RESET}"          ) 
  
  
    ref_file_characterisation =  {
      'target_i0':              target_i0,
      'Wi_target':              Wi_target,
      'Htarget_Rmax':           Htarget_Rmax, 
      'normalisation_factor':   normalisation_factor           
    }
    
    with open( f"{ref_file_characterisation_fname}", "wb") as file:
      pickle.dump(ref_file_characterisation, file, pickle.HIGHEST_PROTOCOL)



  # (1) NORMALISE EVERY OTHER SVS FILE

  display_separator()

  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: will look recursively under:                       {CYAN}{data_source}{RESET} for slide files (files ending with either 'svs' or 'SVS')\n",  flush=True ) 

  slide_file_found  = 0
  is_reference_file = 1                                                                                    # 0=reference file; 1=any other svs file                    
    
  for dir_path, __, files in os.walk( data_source ):                                                                                     # check to see how many svs files remain to be procesed and report

    if not (dir_path==args.data_source):                                                                   # the top level directory (dataset) has to be skipped because it contains the reference svs file    
          
      already_processed_this_slide=False
          
      for f in sorted(files):
      
        current_file = f"{dir_path}/{f}"
    
        if (DEBUG>2):
          print ( f"NORMALISE_STAIN:        INFO: (current_file)                                     {DULL_BLUE}{current_file}{RESET}",    flush=True )
          print ( f"NORMALISE_STAIN:        INFO: (reference_file)                                   {DULL_BLUE}{reference_file}{RESET}",  flush=True )
          # ~ print ( f"NORMALISE_STAIN:        INFO: ( reference_file[-40:])                        {DULL_BLUE}{ reference_file[-40:]}{RESET}",  flush=True )

        if ( f.endswith( 'spcn' )  ):                                                                      # this file has already been handled, so skip
          if (DEBUG>0):
            print ( f"{ORANGE}NORMALISE_STAIN:        INFO: in dir_path {BOLD}{CYAN}{dir_path}{RESET}",  flush=True )
            print ( f"{ORANGE}NORMALISE_STAIN:        INFO: found file  {BOLD}{CYAN}{current_file}{RESET}",  flush=True )
            print ( f"{BOLD}{ORANGE}NORMALISE_STAIN:        INFO: see above. A file with extension {BOLD}{CYAN}.spcn{RESET}{BOLD}{ORANGE} already exists, so will skip and move to the next folder{RESET}",  flush=True )
          global_stats()
          already_processed_this_slide=True 
          display_separator()

      for f in sorted(files):

        current_file = f"{dir_path}/{f}"
    
        if ( f.endswith( 'svs' ) )  |  ( f.endswith( 'SVS' )  ):

          if already_processed_this_slide==False:
              
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
              r,  _, _, _   =  run_batch_colornorm  ( is_reference_file, current_file, reference_file,  nstains,  lamb,  dir_path, level, background_correction, target_i0,  Wi_target, Htarget_Rmax, normalisation_factor, config  )
            if (DEBUG>0):
              if r==SUCCESS:
                print ( f"NORMALISE_STAIN:        INFO: colour normalisation complete",  flush=True )
                display_separator()

              else:
                print ( f"NORMALISE_STAIN:        INFO: colour normalisation failed for this slide ... continuing",  flush=True )
                display_separator()

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def global_stats():

  has_svs_file_count   = 0
  has_spcn_file_count  = 0
  has_both_count       = 0

  for dir_path, __, files in os.walk( args.data_source ):

    if not (dir_path==args.data_source):                                                                   # the top level directory (dataset) has to be skipped because it contains the reference svs file    

      has_an_svs_file  = False
          
      for f in sorted( files ):
      
        current_file = f"{dir_path}/{f}"
    
        if ( f.endswith( 'svs' ) )  |  ( f.endswith( 'SVS' )  ):
          has_an_svs_file = True
          has_svs_file_count += 1
          break                                                                                            # there should only be one SVS file per directory, so can break out of the loop

      for f in sorted( files ):

        current_file = f"{dir_path}/{f}"
    
        if  f.endswith( 'spcn' ) :
            has_spcn_file_count += 1
            if has_an_svs_file:
              has_both_count += 1
            break                                                                                          # there should only be one SPCN file per directory, so can break out of the loop            

  if (DEBUG>10):
    print ( f"NORMALISE_STAIN:        INFO: there are         {BOLD}{MIKADO}{has_svs_file_count:3d}{RESET}      svs  files in total in the working dataset",     flush=True )
    print ( f"NORMALISE_STAIN:        INFO: there are         {BOLD}{MIKADO}{has_spcn_file_count:3d}{RESET}      spcn files in total in the working dataset",    flush=True )

  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: of the total {BOLD}{MIKADO}{has_svs_file_count:3d}{RESET} svs files, {BOLD}{MIKADO}{has_both_count:3d}{RESET} have already been (spcn) stain normalised and {BOLD}{MIKADO}{has_svs_file_count-has_spcn_file_count:3d}{RESET} cases remain to be stain normalised",          flush=True )


def display_separator():
  print( "\n__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n" )


#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_source',                             type=str  )
  p.add_argument('--dataset',                                 type=str  )
  p.add_argument('--reference_file',                          type=str  )
  p.add_argument('--force_reference_file_characterisation',   type=str  )

  args, _ = p.parse_known_args()

  main(args)
      
