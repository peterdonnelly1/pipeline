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
#   4 here are the dependencies:
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
import psutil
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


  num_cpus = multiprocessing.cpu_count()
  pid = os.getpid()
  process   = psutil.Process(pid)
  memoryUse = process.memory_info()[0]/2.**30  # memory use in GB...I think
  affinity  = os.sched_getaffinity(pid)
  
  if DEBUG>0:
    print( f'{SAVE_CURSOR}{CLEAR_LINE}{RESET}  status {MIKADO}{process.status()}{RESET}  affinity {MIKADO}{affinity}{RESET}  pid {MIKADO}{pid:>6d}{RESET}   memory use: {MIKADO}{100*memoryUse:3.1f}{RESET}%   {CLEAR_LINE}{RESTORE_CURSOR}')
  
  # added this in Jun 2022 because my AMD computer started using only one of the 32 available CPUs
  # apparently others have had this issue:see e.g. https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
  x = {i for i in range(num_cpus)}
  os.sched_setaffinity( pid, x)

  data_dir       = args.data_dir
  reference_file = f"{data_dir}/{args.reference_file}"
  
  # ~ reference_file = f"{data_dir}/TCGA-IP-7968-11A-01-TS1.aa84dfd6-6660-4488-b7d6-7652445a6f35.svs"

  if os.path.isfile(reference_file)!=True:
    print ( f"{RED}CLASSI:        FATAL:  the image reference file you provided ('{CYAN}{reference_file}{RESET}{RED}') does not exist.{RESET}" )
    print ( f"{RED}CLASSI:        FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)


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

  display_separator()

  if (DEBUG>0):
    print ( f"NORMALISE_STAIN:        INFO: will look recursively under:                       {MAGENTA}{data_dir}{RESET} for slide files (files ending with either 'svs' or 'SVS')\n",  flush=True ) 

  slide_file_found  = 0
  is_reference_file = 1
  
    
  for dir_path, __, files in os.walk( data_dir):

    if not (dir_path==args.data_dir):                                                                      # the top level directory (dataset) has to be skipped because it contains the reference svs file    
          
      already_processed_this_slide=False
          
      
      for f in sorted(files):
      
        current_file = f"{dir_path}/{f}"
    
        if (DEBUG>2):
          print ( f"NORMALISE_STAIN:        INFO: (current_file)                                     {DULL_BLUE}{current_file}{RESET}",    flush=True )
          print ( f"NORMALISE_STAIN:        INFO: (reference_file)                                   {DULL_BLUE}{reference_file}{RESET}",  flush=True )
          # ~ print ( f"NORMALISE_STAIN:        INFO: ( reference_file[-40:])                        {DULL_BLUE}{ reference_file[-40:]}{RESET}",  flush=True )

        if ( f.endswith( 'spcn' )  ):                                                                      # this folder has already been handled, so set a flag
          if (DEBUG>0):
            print ( f"{ORANGE}NORMALISE_STAIN:        INFO: a file with extension {CYAN}spcn{RESET}{ORANGE} exists in this folder, so will move on to the next folder",  flush=True )
          already_processed_this_slide=True 


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
            


def display_separator():
  print( "\n__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n" )


#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                      type=str, default="/home/peter/git/pipeline/working_data" )
  p.add_argument('--reference_file',                type=str                                                  )
  
  args, _ = p.parse_known_args()

  main(args)
      
