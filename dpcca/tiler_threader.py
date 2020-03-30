
import os
#from threading import Thread
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures import ALL_COMPLETED


from tiler_scheduler import tiler_scheduler
from tiler import tiler

DEBUG=0
SUCCESS=True
FG1="\033[38;5;190m "
RESET="\033[m"

def tiler_threader( args, n_samples, n_tiles, stain_norm, norm_method ):

  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just passing on one of each, passed in as the parameters above

  num_cpus = multiprocessing.cpu_count()

  if (DEBUG>0):
    print ( f"TILER_THREADER:          INFO: about to launch {FG1}{num_cpus}{RESET} threads", flush=True )

  executor = ProcessPoolExecutor(max_workers=num_cpus)
  
  tasks = []
  for n in range(0,num_cpus):
    task=executor.submit( tiler_scheduler, args, n_samples, n_tiles, stain_norm, norm_method, n, num_cpus)
    tasks.append(task)
 
  results = [fut.result() for fut in wait(tasks).done]
 
  return SUCCESS
