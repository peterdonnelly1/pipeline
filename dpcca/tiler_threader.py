
import os
import sys
import time
import numpy as np

#from threading import Thread
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures import ALL_COMPLETED


from tiler_scheduler import tiler_scheduler
from tiler import tiler

WHITE='\033[37;1m'
DIM_WHITE='\033[37;2m'
CYAN='\033[36;1m'
RED='\033[31;1m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GREEN='\033[32;1m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1

SUCCESS=True
FG1="\033[38;5;190m "
RESET="\033[m"

def tiler_threader( args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method ):

  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just passing on one of each, passed in as the parameters above
  just_test = args.just_test

  if just_test=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set. To produce a 2D contiguous output, ALL tiles will be used including background & degenerate tiles (tile statistics are valid, but will show all tiles as 'ok'){RESET}" ) 
        
  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just passing on one of each, passed in as the parameters above
  just_profile=args.just_profile
  
  if just_profile=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_profile' flag is set. Will display slide/tile profiles and then exit{RESET}" )
    
  num_cpus = multiprocessing.cpu_count()


  # (1) First, make sure there are enough samples available to cover the user's requested "n_samples"

  class_file_count = 0
      
  for dir_path, dirs, file_names in os.walk( args.data_dir ):
  
    for d in dirs:
  
      cname = os.path.join(dir_path, d, args.class_numpy_file_name)
          
      if os.path.isfile(cname):
        class_file_count +=1
          
  if class_file_count<np.max(args.n_samples)+1:
    print( f"{RED}TILER_THREADER: FATAL: There aren't enough samples. A file count just now (using 'class.npy' files as a proxy) shows there are {CYAN}{class_file_count}{RESET}{RED} samples, whereas the largest value provided in 'n_samples' = {CYAN}{np.max(args.n_samples)}{RESET}{RED} ... halting now{RESET}" ) 
    sys.exit(0)   
  else:
    print( f"TILER_THREADER: INFO: \033[1ma file count just now (using 'class.npy' files as a proxy) shows that there are enough samples ({class_file_count}) to perform all runs (configured n_samples = {args.n_samples})\033[m" ) 
    

  # (2) Then launch an appropriate number of 'tiler_scheduler' processes

  executor = ProcessPoolExecutor(max_workers=num_cpus)
  tasks = []

  if just_test=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set. Only one process will be used (to ensure the same tiles aren't selected over and over){RESET}" )     
    task=executor.submit( tiler_scheduler, args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, 0, 1 )  
    tasks.append(task)
  else:
    if (DEBUG>0):
      print ( f"TILER_THREADER: INFO: about to launch {FG1}{num_cpus}{RESET} threads", flush=True )    
    for n in range(0,num_cpus):
      task=executor.submit( tiler_scheduler, args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, n, num_cpus)
      tasks.append(task)
      
  results = [fut.result() for fut in wait(tasks).done]

  return SUCCESS
