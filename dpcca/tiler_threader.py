
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
ORANGE='\033[38;2;204;85;0m'
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

SUCCESS=True

DEBUG=1


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
          
  if class_file_count<np.max(args.n_samples):
    print( f"{RED}TILER_THREADER: FATAL: There aren't enough samples. A file count just now (using 'class.npy' files as a proxy) shows there are at most {MIKADO}{class_file_count}{RESET}{RED} samples, whereas (the largest value in) user configuation parameter'n_samples' = {MIKADO}{np.max(args.n_samples)}{RESET}{RED} ... halting now{RESET}" ) 
    sys.exit(0)   
  else:
    print( f"TILER_THREADER: INFO: {WHITE}a file count just now (using '{MAGENTA}class.npy{RESET}' files as a proxy) shows that there are enough image samples ({MIKADO}{class_file_count}{RESET}{WHITE}) to perform all requested runs (configured n_samples is {MIKADO}{args.n_samples}{RESET})\033[m" ) 
    

  # (2) Then launch an appropriate number of 'tiler_scheduler' processes

  executor = ProcessPoolExecutor(max_workers=num_cpus)
  tasks = []

  if just_test=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set. Only one process will be used (to ensure the same tiles aren't selected over and over){RESET}" )     
    task=executor.submit( tiler_scheduler, args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, 0, 1 )  
    tasks.append(task)
  else:
    if (DEBUG>0):
      print ( f"TILER_THREADER: INFO: about to launch {MIKADO}{num_cpus}{RESET} tiler_scheduler threads", flush=True )    
    for n in range(0,num_cpus):
      task=executor.submit( tiler_scheduler, args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, n, num_cpus)
      tasks.append(task)
      
  results = [fut.result() for fut in wait(tasks).done]

  return SUCCESS
