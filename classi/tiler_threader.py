import os
import sys
import math
import time
import pplog
import random
import signal
import psutil

import numpy as np
import multiprocessing

#from threading import Thread
import multiprocessing
from multiprocessing    import Process, Value
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures import as_completed
from concurrent.futures import ALL_COMPLETED

from tiler_scheduler import tiler_scheduler
from tiler import tiler

from constants  import *

SUCCESS=True

DEBUG=1

a=random.choice( range(50,225) )
b=random.choice( range(50,225) )
c=random.choice( range(50,225) )

BB=f"\033[38;2;{a};{b};{c}m"
  
def tiler_threader( args, flag, count, tiles_needed_per_subtype, n_classes, n_samples, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob ):
  
  num_cpus = multiprocessing.cpu_count()

  start_column = 112
  start_row    = 60-num_cpus
  
  # ~ if num_cpus<13:
    # ~ for r in range ( start_row-28, start_row+20 ):
      # ~ print(f"\033[{r};0f{CLEAR_LINE}", end="", flush=True )  
  
  
  random_array = [ random.random() for i in range(1, len(zoom_out_prob)+1 ) ]
  r_norm       = [ i/(sum(random_array)) for i in random_array ]                                           # make the vector add up to 1

  pplog.log_section( "run", f"      system generated zoom out probabilities vector for selection of tiles (corresponds to zoom out mags) = {r_norm}")

  # DON'T USE args.n_tiles since it is the job level array of numbers of tiles. Rather, used the passed in parameter 'n_tiles' which is the value for this run
  just_test    = args.just_test
  multimode    = args.multimode
  just_profile  =args.just_profile

  if ( ( just_test=='True')  & ( multimode!='True' ) ):
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set (and multimode flag not set). To produce a 2D contiguous output, ALL tiles will be used including background & degenerate tiles (tile statistics are valid, but will show all tiles as 'ok'){RESET}" )         

  if just_profile=='True':
    print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_profile' flag is set. Will display slide/tile profiles and then exit{RESET}" )

  num_cpus     = multiprocessing.cpu_count()
  tiling_tasks = []

  if args.make_balanced=='level_down':
  
    results = tiler_scheduler( args, r_norm, flag, count, tiles_needed_per_subtype, n_classes, n_samples, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob, 0, 1 )
  
  else:
    
    tiling_workers = ProcessPoolExecutor( max_workers=num_cpus )

    if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):
      print( f"{ORANGE}TILER_THREADER: INFO: CAUTION! 'just_test' flag is set (and multimode flag not set). Only one process will be used (to ensure the same tiles aren't selected more than one time){RESET}" )     
      tiling_task=tiling_workers.submit( tiler_scheduler, args, r_norm, flag, count, tiles_needed_per_subtype, n_classes, n_samples, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob, 0, 1 )
      tiling_tasks.append(tiling_task)
  
    else:                                                                                                    # train
      if DEBUG>0:
        print ( f"TILER_THREADER: INFO: about to launch {MIKADO}{num_cpus}{RESET} tiler_scheduler threads{CLEAR_LINE}",  flush=True       )
      if DEBUG>20:
        print( f"TILER_THREADER: INFO: number of image samples = {CARRIBEAN_GREEN}{count}{RESET}",           flush=True       )
        print( f"TILER_THREADER: INFO: tiles per image         = {CARRIBEAN_GREEN}{n_tiles}{RESET}",         flush=True       )
        print( f"TILER_THREADER: INFO: tile_size               = {CARRIBEAN_GREEN}{tile_size}{RESET}",       flush=True       )
        print( f"TILER_THREADER: INFO: batch_size              = {CARRIBEAN_GREEN}{batch_size}{RESET}",      flush=True       )
  
      for row in range(start_row-1, start_row+num_cpus):
        print ( f"{SAVE_CURSOR}\r\033[{row};0H{CLEAR_LINE}{RESTORE_CURSOR}", end="", flush=True )
  
      for n in range(0, num_cpus):
        tiling_task = tiling_workers.submit( tiler_scheduler, args, r_norm, flag, count, tiles_needed_per_subtype, n_classes, n_samples, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob, n, num_cpus )
        tiling_tasks.append(tiling_task)
  
    wait( tiling_tasks, return_when=ALL_COMPLETED )
  
    
    if ( ( just_test!='True')  | ( multimode=='image_rna' ) ):                                               # training mode or multimode test mode
      
      results = [ tiling_tasks[x].result() for x in range(0, num_cpus) ]
      if sum(results)==0:
        print ( f"{RED}TILER_THREADER:  FATAL:  no tiles at all were successfully processed{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:  possible cause: perhaps you changed to a different cancer type but did not regenerate the dataset?{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:                  if so, use the {CYAN}-r {RESET}{RED}option ('{CYAN}REGEN{RESET}{RED}') to force the dataset to be regenerated into the working directory{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:                  e.g. '{CYAN}./do_all.sh -d <cancer type code> -i image ... {CHARTREUSE}-r True{RESET}{RED}'{RESET}\n\n" )
        time.sleep(10)                                    
        sys.exit(0)   
    else:                                                                                                    # test mode
      results = tiling_tasks[0].result()
      if results==0:
        print ( f"{RED}TILER_THREADER:  FATAL:  no tiles at all were successfully processed{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:  possible cause: perhaps you changed to a different cancer type but did not regenerate the dataset?{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:                  if so, use the {CYAN}-r {RESET}{RED}option ('{CYAN}REGEN{RESET}{RED}') to force the dataset to be regenerated into the working directory{RESET}" )
        print ( f"{RED}TILER_THREADER:  FATAL:                  e.g. '{CYAN}./do_all.sh -d <cancer type code> -i image ... {CHARTREUSE}-r True{RESET}{RED}'{RESET}\n\n" )
        time.sleep(10)                                    
        sys.exit(0)     



  if DEBUG>0:
    if flag == 'UNIMODE_CASE____IMAGE':
      offset=4
    else:
      offset=3
    
    total_slides_processed = results if args.make_balanced=='level_down' else sum(results) if ( ( just_test!='True' ) | ( multimode=='image_rna' ) ) else results
    
    np.set_printoptions(formatter={'int': lambda x: "{:>5d}".format(x)})
    print ( f"{SAVE_CURSOR}\r\033[{start_row-offset};0f{CLEAR_LINE}{RESET}TILER_THREADER: INFO: {CYAN}{flag} \r\033[50Ctotal slides processed: {MIKADO}{total_slides_processed:<6d}; per thread: {MIKADO}{np.array(results)}{RESET}{RESTORE_CURSOR}", flush=True, end=""  )                  
    time.sleep(1)
    
  
  print (f"\033[{start_row+num_cpus+4};0H", end='' )
        
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
