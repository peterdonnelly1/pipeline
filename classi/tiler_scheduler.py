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

from constants  import *

DEBUG=1

    
def tiler_scheduler( args, r_norm, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method, my_thread, num_threads ):
  
  num_cpus = multiprocessing.cpu_count()

  start_column = 170
  start_row    = 60-num_cpus

    
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
  multimode               = args.multimode
  n_samples               = args.n_samples
  
  
  walker     = os.walk( data_dir, topdown=True )

  dir_count=0

  if (DEBUG>12):
    print ( f"TILER_SCHEDULER_{FG3}:         INFO:                          my_thread = {FG4}{my_thread:2d}{RESET}", flush=True ) 

  slides_processed     = 0
  dirs_which_have_flag = 0
  
  if ( ( just_test!='True' ) | ( multimode=='image_rna') ):                                                # training mode or multimode test mode
    my_quota           = -(count//-num_cpus)                                                               #   how many slides each process has to handle
    if count>10:
      my_expanded_quota  = int (1.3 * my_quota)                                                            #   because some threads will be "luckier" than others in coming across slides with the correct flag
    else:
      my_expanded_quota  = int (3.  * my_quota)                                                            #   because some threads will be "luckier" than others in coming across slides with the correct flag
  else:                                                                                                    # test mode
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
            
            if stain_norm=="spcn":                                                                         
              if ( f.endswith( "spcn" ) ):                                                                 # then the stain normalised version of the slide will have extension 'spcn'
                pqn = f"{d}/{f}"
                result = tiler( args, r_norm, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread )
                if result==SUCCESS:
                  slides_processed+=1
                  if ( ( just_test=='True' ) & ( multimode!='image_rna') )  & (my_thread==0):
                    print ( f"{SAVE_CURSOR}\r\033[300C\033[11B{RESET}{CARRIBEAN_GREEN}{slides_processed}/{my_quota}{RESET}{RESTORE_CURSOR}{CLEAR_LINE}", flush=True ) 
                  else:
                    print ( f"{SAVE_CURSOR}\r\033[300C\033[{my_thread}B{RESET}{CARRIBEAN_GREEN}{slides_processed}/{my_quota}{RESET}{RESTORE_CURSOR}{CLEAR_LINE}", flush=True )                           
                  if slides_processed>=my_expanded_quota:
                    break
                else:
                  print(f"{ORANGE}TILER_SCHEDULER_{FG3}: WARNING: not enough qualifying tiles ! Slide will be skipped. {MIKADO}{slides_processed}{RESET}{ORANGE} slides have been processed{RESET}", flush=True)
                  if slides_processed<n_samples:
                    print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    n_samples=slides_processed

            else:                                                                                          # look for and use normal versions of the slides
              if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ) | ( f.endswith( "tif" ) ) | ( f.endswith( "tif" ) )  | ( f.endswith( "TIF" ) ) | ( f.endswith( "TIFF" ) ):
                pqn = f"{d}/{f}"
                result = tiler( args, r_norm, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread )
                if result==SUCCESS:
                  slides_processed+=1
                  if ( ( just_test=='True' ) & ( multimode!='image_rna') )  & (my_thread==0):
                    print ( f"{SAVE_CURSOR}\r\033[300C\033[13B{RESET}{CARRIBEAN_GREEN}{slides_processed}/{my_quota}{RESET}{RESTORE_CURSOR}{CLEAR_LINE}", flush=True ) 
                  else:
                    print ( f"{SAVE_CURSOR}\r\033[300C\033[{my_thread}B{RESET}{CARRIBEAN_GREEN}{slides_processed}/{my_quota}{RESET}{RESTORE_CURSOR}{CLEAR_LINE}", flush=True )                           
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
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{GREEN}thread {MIKADO}{my_thread:2d}{RESET}{GREEN} exiting - on quota  {RESET}", flush=True  )
  elif slides_processed>my_quota:
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{MAGENTA}thread {MIKADO}{my_thread:2d}{RESET}{MAGENTA} exiting - over quota {RESET}", flush=True )
  else:
    print ( f"\033[{start_row+my_thread};{start_column+94}f  {RESET}{RED}thread {MIKADO}{my_thread:2d}{RESET}{RED} exiting - under quota {RESET}", flush=True )


  return(slides_processed)
