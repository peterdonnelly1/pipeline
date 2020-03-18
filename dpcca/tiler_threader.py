
import os
from threading import Thread
import multiprocessing

from tiler_scheduler import tiler_scheduler
from tiler import tiler

DEBUG=1
SUCCESS=False
FG1="\033[38;5;190m "
RESET="\033[m"


def tiler_threader( args, n_required, stain_norm ):

  num_cpus = multiprocessing.cpu_count()

  if (DEBUG>0):
    print ( f"TILER_THREADER:          INFO: about to launch {FG1}{num_cpus}{RESET} threads", flush=True ) 

  threads = []
  for i in range(num_cpus):
    thr = Thread(target=tiler_scheduler, args=(args, n_required, stain_norm, i, num_cpus) )
    threads.append(thr)

  for t in threads:
    t.start()

  for t in threads:
    t.join()
    
  return SUCCESS
  
