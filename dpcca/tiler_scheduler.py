import os
import sys
import numpy as np
import multiprocessing
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

    
def tiler_scheduler( args, r_norm, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method, my_thread, num_threads ):
  
  num_cpus = multiprocessing.cpu_count()

  start_column = 170
  start_row    = 70-num_cpus

    
  np.random.seed(my_thread)
  r=np.random.randint(0,255)
  g=np.random.randint(0,255)
  b=np.random.randint(0,255)
 
  # DON'T USE args.n_samples or args.n_tiles since they are the complete, job level list of samples and numbers of tiles. Here we are just using one of each, passed in as the parameters above
  data_dir                = args.data_dir
  divide_cases            = args.divide_cases
  input_mode              = args.input_mode
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  rna_file_suffix         = args.rna_file_suffix  
  just_test               = args.just_test
  n_samples               = args.n_samples
  
  
  walker     = os.walk( data_dir, topdown=True )

  dir_count=0

  if (DEBUG>12):
    print ( f"TILER_SCHEDULER_{FG3}:         INFO:                          my_thread = {FG4}{my_thread:2d}{RESET}", flush=True ) 

  slides_processed     = 0
  dirs_which_have_flag = 0
  
  if just_test!='True':
    my_quota           = -(count//-num_cpus)                                                               # how many slides each process has to handle
    my_expanded_quota  = int (1.3 * my_quota)                                                              # because some threads will be "luckier" than others in coming across slides with the correct flag
  else:
    my_quota           = count
    my_expanded_quota  = my_quota
  
  if DEBUG>4:
    # ~ if (my_thread>15) & (my_thread<20):
    if (my_thread>18):
      print ( f"\r{RESET}TILER_SCHEDULER_{my_thread:2d}:      INFO:  my_quota          = {MIKADO}{my_quota}{RESET}{CLEAR_LINE}", flush=True ) 
      print ( f"\r{RESET}TILER_SCHEDULER_{my_thread:2d}:      INFO:  my_expanded_quota = {AMETHYST}{my_expanded_quota}{RESET}{CLEAR_LINE}", flush=True ) 
  
  for root, dirs, files in walker:                                                                         # go through all the directories, but only tackle every my_thread'th directory
    
    for d in dirs:

      dir_count+=1
      modulus=dir_count%num_threads

      if not ( modulus==my_thread ):                                                                       # skip over directories that other threads are handling
        pass
      else:
        fqd = f"{root}/{d}"
      
        has_flag=False                                                                    # in this case, all image cases are candidates ('ALL_ELIGIBLE_CASES' aren't flagged as such)
        try:
          fqn = f"{root}/{d}/{flag}"        
          f = open( fqn, 'r' )
          has_flag=True
          dirs_which_have_flag+=1   
          if DEBUG>48:
            print ( f"{GREEN}{flag}{RESET}", flush=True )          
        except Exception:
          if DEBUG>48:
            print ( f"{RED}{flag}{RESET}",   flush=True )       
      
        if has_flag==True:
            
          for f in os.listdir( fqd ):
            
            if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ) | ( f.endswith( "tif" ) ) | ( f.endswith( "tif" ) )  | ( f.endswith( "TIF" ) ) | ( f.endswith( "TIFF" ) ):
              pqn = f"{d}/{f}"
              result = tiler( args, r_norm, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread )
              if result==SUCCESS:
                slides_processed+=1
                if slides_processed>=my_expanded_quota:
                  break
              else:
                print(f"{ORANGE}TILER_SCHEDULER_{FG3}: WARNING: not enough qualifying tiles ! Slide will be skipped. {MIKADO}{slides_processed}{RESET}{ORANGE} slides have been processed{RESET}", flush=True)
                if slides_processed<n_samples:
                  print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                  n_samples=slides_processed

      if slides_processed>=my_expanded_quota:
        break

    if slides_processed>=my_expanded_quota:
      break
                  
  if slides_processed==my_quota:
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{CLEAR_LINE}{GREEN}thread {MIKADO}{my_thread:2d}{RESET}{GREEN} exiting - on quota  {CLEAR_LINE}{RESET}", flush=True  )
  elif slides_processed>my_quota:
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{CLEAR_LINE}{MAGENTA}thread {MIKADO}{my_thread:2d}{RESET}{MAGENTA} exiting - over quota {CLEAR_LINE}{RESET}", flush=True )
  else:
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{CLEAR_LINE}{RED}thread {MIKADO}{my_thread:2d}{RESET}{RED} exiting - under quota {CLEAR_LINE}{RESET}", flush=True )


  return(slides_processed)
