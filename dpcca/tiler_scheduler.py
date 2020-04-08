
import os
import sys

from pathlib import Path
from tiler import tiler

DEBUG=0
FAIL=False
SUCCESS=True
FG3="\033[38;5;86m"
FG4="\033[38;5;150m"
FG5="\033[38;5;210m"
FG6="\033[38;5;220m"
RESET="\033[m"

def tiler_scheduler( args, n_samples, n_tiles, batch_size, stain_norm, norm_method, my_thread, num_threads ):

  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just using one of each, passed in as the parameters above
  data_dir = args.data_dir
  
  walker     = os.walk( data_dir, topdown=True )

  dir_count=0

  if (DEBUG>0):
    print ( f"TILER_SCHEDULER:         INFO:                          my_thread = {FG4}{my_thread:2d}{RESET}", flush=True ) 

  slides_processed = 0
  
  for root, dirs, files in walker:
    for d in dirs:
      dir_count+=1
      modulus=dir_count%num_threads
      # skip over files that other threads are handling
      if not ( modulus==my_thread) :
        pass
      else:
        if (DEBUG>0):
          print ( f"TILER_SCHEDULER_{FG3}{my_thread:2d}:      INFO:  says: 'this one's mine!'  (modulus = {modulus:2d}{RESET})", flush=True ) 
        fqd = f"{root}/{d}"
        if (DEBUG>0):
          print ( f"TILER_SCHEDULER:         INFO:  d             =  {FG4}{d}{RESET}", flush=True ) 
          print ( f"TILER_SCHEDULER:         INFO:  fqd           =  {FG4}{fqd}{RESET}",   flush=True   ) 
        for f in os.listdir(fqd):
          if (DEBUG>0):
            print ( f"TILER_SCHEDULER:         INFO:  f             =  {FG5}{f}{RESET}", flush=True )
          if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ):
            pqn = f"{d}/{f}"
            if (DEBUG>0):
              print ( f"TILER_SCHEDULER:         INFO:  current slide =  {FG6}{f}{RESET}", flush=True ) 
              print ( f"TILER_SCHEDULER:         INFO:  fqn           =  {FG6}{pqn}{RESET}",   flush=True   )
            result = tiler( args, n_tiles, batch_size, stain_norm, norm_method, d, f, my_thread )
            if result==SUCCESS:
              slides_processed+=1
            else:
              sys.exit(0)
              

            if n_samples%num_threads==0:                                                                    # then each thread can do the same number of slides an we will have exactly n_samples slides processed in total                                         
              if slides_processed>=(n_samples//num_threads):                                                
                if (DEBUG>0):
                  print ( f"TILER_SCHEDULER:     INFO:  required number of slides \033[35m{n_samples}\033[m for processor \033[35m{my_thread}\033[m completed, breaking inner loop", flush=True ) 
                return SUCCESS
            else:
              if slides_processed>=(n_samples//num_threads + 1):                                            # then each thread will need to do one extra slide to ensure n_samples is covered
                if (DEBUG>0):
                  print ( f"TILER_SCHEDULER:     INFO:  required number of slides \033[35m{n_samples}\033[m for processor \033[35m{my_thread}\033[m completed, breaking inner loop", flush=True ) 
                return SUCCESS

          else:
            pass
  
  return FAIL
  
