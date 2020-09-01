
import os
import sys

from pathlib import Path
from tiler import tiler

FAIL=False
SUCCESS=True
FG3="\033[38;5;86m"
FG4="\033[38;5;150m"
FG5="\033[38;5;210m"
FG6="\033[38;5;220m"

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

DEBUG=0

def tiler_scheduler( args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, my_thread, num_threads ):

  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just using one of each, passed in as the parameters above
  data_dir                = args.data_dir
  input_mode              = args.input_mode
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  rna_file_suffix         = args.rna_file_suffix
  
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
          print ( f"TILER_SCHEDULER:         INFO:  fqd/d          =  \r\033[49C{FG4}{fqd}{RESET}\r\033[122C| \r\033[{124+6*(int(d[0],16))}C{FG4}{d}{RESET}", flush=True ) 
          #print ( f"TILER_SCHEDULER:         INFO:  fqd           =  {FG4}{fqd}{RESET}",   flush=True   )
          
        for f in os.listdir(fqd):
          
          if (DEBUG>0):
            print ( f"TILER_SCHEDULER:         INFO:  f             =  {FG5}{f}{RESET}", flush=True )
          if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ):
            pqn = f"{d}/{f}"
            if (DEBUG>0):
              print ( f"TILER_SCHEDULER:         INFO:  current slide =  {FG6}{f}{RESET}", flush=True ) 
              print ( f"TILER_SCHEDULER:         INFO:  fqn           =  {FG6}{pqn}{RESET}",   flush=True   )
            result = tiler( args, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread )
            if result==SUCCESS:
              slides_processed+=1
            else:
              print(f"{ORANGE}TILER_SCHEDULER: WARNING: slide skipped, therefore reducing 'n_samples' from {CYAN}{n_samples}{RESET} to {CYAN}{n_samples-1}{RESET}", flush=True)
              n_samples -= 1
              if n_samples<1:
                print( f"{RED}TILER_SCHEDULER: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                sys.exit(0)
                

            if n_samples%num_threads==0:                                                                    # then each thread can do the same number of slides an we will have exactly n_samples slides processed in total                                         
              if slides_processed>=(n_samples//num_threads + 1):                                                
                if (DEBUG>0):
                  print ( f"{GREEN}TILER_SCHEDULER:     INFO:  required number of slides {RESET}({CYAN}{n_samples//num_threads+1}{RESET}){GREEN} for processor {RESET}({CYAN}{my_thread}{RESET}){GREEN} completed, breaking inner loop{RESET}", flush=True ) 
                return SUCCESS
            else:
              if slides_processed>=(n_samples//num_threads + 2):                                            # then each thread will need to do one extra slide to ensure n_samples is covered
                if (DEBUG>0):
                  print ( f"{GREEN}TILER_SCHEDULER:     INFO:  required number of slides {RESET}({CYAN}{n_samples//num_threads+2}{RESET}){GREEN} for processor {RESET}({CYAN}{my_thread}{RESET}){GREEN} completed, breaking inner loop{RESET}", flush=True ) 
                return SUCCESS

          else:
            pass
  
  return FAIL
