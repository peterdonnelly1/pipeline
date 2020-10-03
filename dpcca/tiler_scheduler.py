
import os
import sys
import multiprocessing

import numpy as np
from pathlib import Path
from tiler import tiler

FAIL=False
SUCCESS=True

FG3="\033[38;5;100m"
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

DEBUG=1

num_cpus = multiprocessing.cpu_count()

def tiler_scheduler( args, n_samples, n_tiles, tile_size, batch_size, stain_norm, norm_method, my_thread, num_threads ):

  np.random.seed(my_thread)
  r=np.random.randint(0,255)
  g=np.random.randint(0,255)
  b=np.random.randint(0,255)
 
  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just using one of each, passed in as the parameters above
  data_dir                = args.data_dir
  input_mode              = args.input_mode
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  rna_file_suffix         = args.rna_file_suffix
  
  walker     = os.walk( data_dir, topdown=True )

  dir_count=0

  if (DEBUG>1):
    print ( f"TILER_SCHEDULER_{FG3}:         INFO:                          my_thread = {FG4}{my_thread:2d}{RESET}", flush=True ) 

  slides_processed = 0
  
  for root, dirs, files in walker:                                                                         # go through all the directories, but only tackle every my_thread'th directory
    
    for d in dirs:

      dir_count+=1
      modulus=dir_count%num_threads

      if not ( modulus==my_thread ):                                                                            # skip over directories that other threads are handling
        pass
      else:
        if (DEBUG>1):
          print ( f"TILER_SCHEDULER_{FG3}{my_thread:2d}:      INFO:  says: 'this one's mine!'  (modulus = {modulus:2d}{RESET})", flush=True ) 
        fqd = f"{root}/{d}"
        if (DEBUG>1):
          print ( f"TILER_SCHEDULER_{FG3}:         INFO:  fqd/d          =  \r\033[49C{FG4}{fqd}{RESET}\r\033[122C| \r\033[{124+6*(int(d[0],16))}C{FG4}{d}{RESET}", flush=True ) 
          #print ( f"TILER_SCHEDULER:         INFO:  fqd           =  {FG4}{fqd}{RESET}",   flush=True   )
          
        for f in os.listdir( fqd ):
          
          if (DEBUG>1):
            print ( f"TILER_SCHEDULER_{FG3}:         INFO:  f             =  {FG5}{f}{RESET}", flush=True )
          if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ) | ( f.endswith( "tif" ) ) | ( f.endswith( "tif" ) )  | ( f.endswith( "TIF" ) ) | ( f.endswith( "TIFF" ) ):
            pqn = f"{d}/{f}"
            if (DEBUG>1):
              print ( f"TILER_SCHEDULER_{FG3}:         INFO:  current slide =  {FG6}{f}{RESET}", flush=True ) 
              print ( f"TILER_SCHEDULER_{FG3}:         INFO:  fqn           =  {FG6}{pqn}{RESET}",   flush=True   )
            result = tiler( args, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread )
            if result==SUCCESS:
              slides_processed+=1
              if DEBUG>7:
                print ( f"TILER_SCHEDULER_\033[38;2;{r};{g};{b}m{my_thread:2d}:     INFO:  \033[{3*slides_processed}Cslides_processed = {slides_processed}{RESET}", flush=True )
            else:
              print(f"{ORANGE}TILER_SCHEDULER_{FG3}: WARNING: not enough qualifying tiles ! Slide will be skipped. {MIKADO}{slides_processed}{RESET}{ORANGE} slides have been processed{RESET}", flush=True)
              if slides_processed<n_samples:
                print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                n_samples=slides_processed

          else:                                                                                             # not an image files
            pass

      # check to see if tiler_threader has set the "STOP" flag
      fq_name = f"{data_dir}/SUFFICIENT_SLIDES_TILED"

      start_column = 180
      start_row = 67-num_cpus-1
      try:
        f = open( fq_name, 'r' )
        if (DEBUG>0):
          print ( f"\033[{start_row+my_thread};{start_column}f  {RESET}'{CYAN}SUFFICIENT_SLIDES_TILED{RESET}' flag seen - thread {MIKADO}{my_thread}{RESET} will now exit{CLEAR_LINE}{RESET}", flush=True ) 
        sys.exit(0)
      except Exception:
        pass
      

  if (DEBUG>2):
    print ( f"TILER_SCHEDULER_\033[38;2;{r};{g};{b}m{my_thread:2d}:     INFO:  \r\033[150C processed                 {RESET}{MIKADO}{slides_processed}{RESET} slides for CPU {MIKADO}{my_thread:2d}{RESET}           ... returning from thread{RESET}", flush=True ) 
  
  return SUCCESS
