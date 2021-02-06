import os
import sys
import math
import time
import random
import signal
import psutil
import numpy as np
import multiprocessing

#from threading import Thread
import multiprocessing
from multiprocessing import Process, Value
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures import as_completed
from concurrent.futures import ALL_COMPLETED

from tiler_scheduler import tiler_scheduler
from tiler import tiler

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
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;task192;203m'
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

a=random.choice( range(50,225) )
b=random.choice( range(50,225) )
c=random.choice( range(50,225) )

BB=f"\033[38;2;{a};{b};{c}m"
  
def tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method ):

  num_cpus = multiprocessing.cpu_count()

  start_column = 170
  start_row    = 70-num_cpus
  

  # DON'T USE args.n_tiles since it is the job level array of numbers of tiles. Rather, used the passed in parameter 'n_tiles' which is the value for this run
  just_test    = args.just_test
  just_profile  =args.just_profile

  if just_test=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set. To produce a 2D contiguous output, ALL tiles will be used including background & degenerate tiles (tile statistics are valid, but will show all tiles as 'ok'){RESET}" )         

  if just_profile=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_profile' flag is set. Will display slide/tile profiles and then exit{RESET}" )

  # configure an appropriate number of 'tiler_scheduler' processes    
  num_cpus = multiprocessing.cpu_count()
  executor = ProcessPoolExecutor(max_workers=num_cpus)
  tasks = []


  if just_test=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set. Only one process will be used (to ensure the same tiles aren't selected repeatedly){RESET}" )     
    task=executor.submit( tiler_scheduler, args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method, 0, 1)
    tasks.append(task)
  else:
    if DEBUG>8:
      print ( f"TILER_THREADER: INFO: about to launch {MIKADO}{num_cpus}{RESET} tiler_scheduler threads", flush=True )
    if DEBUG>28:
      print( f"TILER_THREADER: INFO: count                   = {CARRIBEAN_GREEN}{count}{RESET}"           )
      print( f"TILER_THREADER: INFO: n_tiles                 = {CARRIBEAN_GREEN}{n_tiles}{RESET}"         )
      print( f"TILER_THREADER: INFO: batch_size              = {CARRIBEAN_GREEN}{batch_size}{RESET}"      )

    for n in range(0, num_cpus):
      task=executor.submit( tiler_scheduler, args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method, n, num_cpus )
      tasks.append(task)

    wait( tasks, return_when=ALL_COMPLETED )
    
    results = [ tasks[x].result() for x in range(0, num_cpus) ]
     

    if DEBUG>0:
        print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[{start_row+num_cpus};{start_column+3}fALL THREADS HAVE FINISHED{RESET}{RESTORE_CURSOR}", flush=True, end=""  )                     


    if DEBUG>0:
        print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[{start_row+num_cpus+1};{start_column+3}ftotal slides_processed       = {MIKADO}{sum(results)}{RESET}{RESTORE_CURSOR}", flush=True, end=""  )                  
        print ( f"{SAVE_CURSOR}{RESET}{CARRIBEAN_GREEN}\r\033[{start_row+num_cpus+2};{start_column+3}fslides_processed per process = {MIKADO}{results}{RESET}{RESTORE_CURSOR}", flush=True, end=""  )                  
 
        
  return SUCCESS




#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)
