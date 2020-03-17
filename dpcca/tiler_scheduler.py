
import os

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

def tiler_scheduler( args, n_required, my_thread, num_threads ):
  
  data_dir = args.data_dir
  walker     = os.walk( data_dir, topdown=True )

  dir_count = 0

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
            result = tiler( args, d, f, my_thread )
            if result==True:
              slides_processed+=1

            if slides_processed>=n_required:
              if (DEBUG>0):
                print ( f"TILER_SCHEDULER:     INFO:  required amount \033[35m{n_required}\033[m processed, breaking inner loop", flush=True ) 
              return SUCCESS
          else:
            pass
  
  return FAIL
  
