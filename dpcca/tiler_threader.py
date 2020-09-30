
import os
import sys
import math
import time
import signal
import psutil
import numpy as np

#from threading import Thread
import multiprocessing
from multiprocessing import Process, Value
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

  # Launch an appropriate number of 'tiler_scheduler' processes    
  num_cpus = multiprocessing.cpu_count()
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
      
  #results = [fut.result() for fut in wait(tasks).done]

  # periodically check to see if enough samples have been processed by counting the flags each worker has left behind in the directories of the SVS/TIF files it has processed

  if just_test=='False':
    rounded_up_number_required = np.max(args.n_samples)
#    rounded_up_number_required = math.ceil( np.max(args.n_samples) / num_cpus ) * num_cpus
  else:
    rounded_up_number_required = np.max(args.n_samples)

  if (DEBUG>0):
    print ( f"{RESET}TILER_THREADER: INFO: number of slides required, rounded up to be an exact multiple of the number of available CPUs = {MIKADO}{rounded_up_number_required}{RESET}", flush=True )  
  
  sufficient_slides_tiled=False  
  while sufficient_slides_tiled==False:
    
    slides_tiled_count   = 0
    for dir_path, dirs, files in os.walk( args.data_dir ):
  
      if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) has be skipped because it only contains sub-directories, not data   
                    
        for f in files:       

          if f == "SLIDE_TILED_FLAG":
            slides_tiled_count +=1
          
          if slides_tiled_count>=rounded_up_number_required:
            sufficient_slides_tiled=True
 
            # having tiled all the samples needed, set up a flag to tell the workers to exit
            fq_name = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
          
            with open(fq_name, 'w') as f:
              f.write( f"flag file to indicate that we now have enough tiled image files and that workers should now exit" )
              f.close    
              if (DEBUG>0):                
                print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[68;110f  sufficient slides ({MIKADO}{slides_tiled_count}{RESET}{CARRIBEAN_GREEN}) have now been tiled{RESET}{RESTORE_CURSOR}", flush=True, end="" )
              time.sleep(9)
              return SUCCESS

    if just_test=='False':
      time.sleep(1)

    if (DEBUG>0):
      if just_test=='False':
        print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[68;110f  total slides processed so far = {MIKADO}{slides_tiled_count+1}{RESET}", end="" )                     
      else:
        print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[68;110f  total slides processed so far = {MIKADO}{slides_tiled_count+1}{RESET}{RESTORE_CURSOR}", flush=True, end="" )     

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)  
