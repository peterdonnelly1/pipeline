import os
import sys
import random
import psutil
import time
import numpy as np
import multiprocessing
import multiprocessing
from pathlib import Path
import pyvips

from tiler   import tiler

SUCCESS=1
FAIL=0

INSUFFICIENT_TILES=2
INSUFFICIENT_QUALIFYING_TILES=3
MISSING_IMAGE_FILE=4
EXCLUDED_CLASS=5

FG3="\033[38;5;100m"
FG4="\033[38;5;150m"
FG5="\033[38;5;210m"
FG6="\033[38;5;220m"

from constants  import *

DEBUG=1

    
def tiler_scheduler( args, r_norm, flag, slide_count, n_samples, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob, my_thread, num_threads ):

  
  num_cpus = multiprocessing.cpu_count()

  start_column = 112
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
  all_tiles_from_origin   = args.all_tiles_from_origin
  
  
  walker     = os.walk( data_dir, topdown=True )

  dir_count=0

  if (DEBUG>1):
    print ( f"TILER_SCHEDULER_{FG3}:         INFO:                          my_thread = {FG4}{my_thread:2d}{RESET}     ", flush=True ) 
    print ( f"TILER_SCHEDULER_{FG3}:         INFO:                          slide_count     = {FG4}{slide_count:2d}{RESET}         ",     flush=True ) 

  slides_processed     = 0
  dirs_which_have_flag = 0
  

  if args.just_test=='True':
    my_slide_quota              = n_samples if n_samples < slide_count else slide_count
    my_expanded_slide_quota     = my_slide_quota 
  else:
    my_slide_quota             = -(slide_count//-num_cpus)                                                   # how many slides each process has to handle
    my_expanded_slide_quota    = 3*my_slide_quota                                                            # because some threads will be "luckier" than others in coming across slides with the correct flag
                                                                                                           # ('my_expanded_slide_quota' is an internal artifact: only 'my_slide_quota' is shown to the user)
                                                                                                           # it doesn't increase tiling time, just makes sure that each thread looks at every example that it is allocated

  
  if DEBUG>0:
    print ( f"{SAVE_CURSOR}{RESET}\r\033[{start_row-7};0HTILER_SCHEDULER_thread_{PINK}{my_thread:02d}{RESET}:      INFO:  unadjusted tiles/slide         = {MIKADO}{n_tiles}{RESET}{CLEAR_LINE}{RESTORE_CURSOR}",         flush=True ) 
    print ( f"{SAVE_CURSOR}{RESET}\r\033[{start_row-6};0HTILER_SCHEDULER_thread_{PINK}{my_thread:02d}{RESET}:      INFO:  qualifying slides count        = {MIKADO}{slide_count}{RESET}{CLEAR_LINE}{RESTORE_CURSOR}",     flush=True ) 
    print ( f"{SAVE_CURSOR}{RESET}\r\033[{start_row-5};0HTILER_SCHEDULER_thread_{PINK}{my_thread:02d}{RESET}:      INFO:  thread's slide quota           = {MIKADO}{my_slide_quota}{RESET}{CLEAR_LINE}{RESTORE_CURSOR}",  flush=True ) 

  cumulative_tiles_processed = 0
  
  for root, dirs, files in walker:                                                                         # go through all the directories, but only tackle every my_thread'th directory
    
    for d in dirs:

      dir_count+=1
      modulus=dir_count%num_threads

      if not ( modulus==my_thread ):                                                                       # skip over directories that other threads are handling
        pass
      else:
        fqd = f"{root}/{d}"
      
        has_flag=False                                                                                     # in this case, all image cases are candidates ('ALL_ELIGIBLE_CASES' aren't flagged as such)
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
            
            r = random.randint(1, 12)
            s = "s"
            
            if stain_norm=="spcn":
              
              if ( f.endswith( "spcn" ) ):                                                                 # then the stain normalised version of the slide will have extension 'spcn'
                
                pqn = f"{d}/{f}"
                
                tiles_processed, result = tiler( args, r_norm, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method,  zoom_out_mags, zoom_out_prob, d, f, my_thread, r )
                cumulative_tiles_processed += tiles_processed
                
                if result==SUCCESS:
                  slides_processed+=1
                  if ( ( just_test=='True' ) & ( multimode!='image_rna') )  & (my_thread==0):
                    if DEBUG>0:
                      print ( f"{SAVE_CURSOR}\033[{start_row};292H{RESET}{CARRIBEAN_GREEN}{slides_processed:3d} slide{s if slides_processed>1 else ' '} done (quota {my_slide_quota}){RESET}{CLEAR_LINE}{RESTORE_CURSOR}", flush=True ) 
                  else:
                    if DEBUG>0:
                      print ( f"{SAVE_CURSOR}\033[{start_row+my_thread};292H{RESET}{CARRIBEAN_GREEN}{slides_processed} slide{s if slides_processed>1 else ' '} done (quota {my_slide_quota}){RESET}{CLEAR_LINE}{RESTORE_CURSOR}S", flush=True )                           
                  if slides_processed>=my_expanded_slide_quota:
                    break
                elif result==INSUFFICIENT_TILES:
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{ORANGE}{CLEAR_LINE}TILER_SCHEDULER_{FG3}{my_thread}: WARNING: it would not be possible to extract the required number of tiles from this slide, even if they all qualified ! Slide will be skipped. {MIKADO}{slides_processed}{RESET}{ORANGE} slides have been processed{RESET}{RESTORE_CURSOR}", flush=True)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed
                elif result==INSUFFICIENT_QUALIFYING_TILES:
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{ORANGE}{CLEAR_LINE}TILER_SCHEDULER_{FG3}{my_thread}: WARNING: not enough qualifying tiles ! Slide will be skipped. {MIKADO}{slides_processed}{RESET}{ORANGE} slides have been processed{RESET}{RESTORE_CURSOR}", flush=True)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed
                elif result==MISSING_IMAGE_FILE:
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{BOLD_ORANGE}{CLEAR_LINE}TILER: WARNING{my_thread}: there was no spcn file for this case !!! {BOLD_CYAN}{pqn}{RESET}{BOLD_ORANGE}. Skipping and moving to next case{RESET}{RESTORE_CURSOR}", flush=True)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed

            else:                                                                                          # look for and use normal versions of the slides
                                                                                                           
              if ( f.endswith( "svs" ) ) | ( f.endswith( "SVS" ) ) | ( f.endswith( "tif" ) ) | ( f.endswith( "tif" ) )  | ( f.endswith( "TIF" ) ) | ( f.endswith( "TIFF" ) ):

                pqn = f"{d}/{f}"

                if  ( f.endswith( "jpg" ) ) | ( f.endswith( "jpeg" ) ):                                    # openslide can't handle jpg as a source image type, so we need to convert to tif

                  if DEBUG>5:
                    print ( f"{BOLD}{ORANGE}TILER_SCHEDULER_{FG3}{num_cpus}:         INFO:  converting jpeg image source file to tif so that openslide can handle it{RESET}", flush=True )

                  fqn = f"{data_dir}/{d}/{f}"
                  image = pyvips.Image.new_from_file(fqn)

                  f  = f"{f}.tif"                                                                          # point f to the tif file
                  new_fqn = f"{data_dir}/{d}/{f}"
                  image.write_to_file(new_fqn,  tile=True)                                                 # save as tif. Has to be 'tile=True' (nothing to do with CLASSI tiling) or else Openslide won't accept it


                if DEBUG>0:
                  if n_tiles==0:
                    print(f'{SAVE_CURSOR}{RESET}\033[84;0H{BOLD_AMETHYST}n_tiles==0.  This should not be possible{RESET}{RESTORE_CURSOR}', flush=True) 
                     
                
                tiles_processed, result = tiler( args, r_norm, n_tiles, top_up_factors, tile_size, batch_size, stain_norm, norm_method, zoom_out_mags, zoom_out_prob, d, f, my_thread, r )
                cumulative_tiles_processed += tiles_processed

                a = random.choice( range(150+2*my_thread,255) )
                b = random.choice( range(50,225) )
                c = random.choice( range(50,225) )
                BB="\033[38;2;{:};{:};{:}m".format( a,b,c )  
                
                if result==SUCCESS:
                  slides_processed+=1
                  if ( ( just_test=='True' ) & ( multimode!='image_rna') )  & (my_thread==0):
                    if DEBUG>0:
                      print ( f"{SAVE_CURSOR}\033[{start_row};292H{RESET}{CARRIBEAN_GREEN}{slides_processed:3d} slide{s if slides_processed>1 else ' '} done (quota {my_slide_quota}){RESET}{CLEAR_LINE}{RESTORE_CURSOR}", flush=True ) 
                  else:
                    if DEBUG>0:
                      print ( f"{SAVE_CURSOR}\033[{start_row+my_thread};292H{RESET}{CARRIBEAN_GREEN}{slides_processed:3d} slide{s if slides_processed>1 else ' '} done (quota {my_slide_quota}){RESET}{CLEAR_LINE}{RESTORE_CURSOR}S", flush=True )                           
                  if slides_processed>=my_expanded_slide_quota:
                    break              
                elif result==INSUFFICIENT_TILES:
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{BOLD_ORANGE}{CLEAR_LINE}TILER_SCHEDULER_{FG3}{my_thread}: {BB}WARNING: it would not be possible to extract the required number of tiles from this slide, even if they all qualified ! ({BOLD_CYAN}{pqn}{RESET}{BOLD}{BB}). Slide will be skipped. {RESTORE_CURSOR}", flush=True)
                  time.sleep(1)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed
                elif result==INSUFFICIENT_QUALIFYING_TILES:
                  time.sleep(1)
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{BOLD_ORANGE}{CLEAR_LINE}TILER_SCHEDULER_{FG3}{my_thread}: {BB}WARNING: not enough qualifying tiles for this case ! ({BOLD_CYAN}{pqn}{RESET}{BOLD}{BB}). Slide will be skipped. {RESET}{RESTORE_CURSOR}", flush=True)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed
                elif result==MISSING_IMAGE_FILE:
                  time.sleep(1)
                  print(f"{SAVE_CURSOR}{RESET}\033[{start_row+num_cpus};0H{BOLD_ORANGE}{CLEAR_LINE}TILER_SCHEDULER_{FG3}{my_thread}: {BB}WARNING: there was no image file for this case! ({BOLD_CYAN}{pqn}{RESET}{BOLD}{BB}). Slide will be skipped. {RESET}{RESTORE_CURSOR}", flush=True)
                  # ~ if slides_processed<n_samples:
                    # ~ print( f"{RED}TILER_SCHEDULER_{FG3}: FATAL:  n_samples has been reduced to {CYAN}{n_samples}{RESET}{RED} ... halting{RESET}" )
                    # ~ n_samples=slides_processed
                # ~ elif result==EXCLUDED_CLASS:
                  # ~ pass
                else:
                  print (f"{SAVE_CURSOR}{RESET}\033[{start_row+my_thread};270H{BOLD_RED}{result}  <<< unknown error{RESET}{RESTORE_CURSOR}",flush=True )                    
                  time.sleep(1)
                  
                  
                if DEBUG>0:
                  r = f'{RED}FAIL' if result==0 else f'{GREEN}SUCCESS{RESET}' if result==1 else f'{ORANGE}INSUF_TILES{RESET}' if result==2 else f'{ORANGE}INSUF_QUALIFYING{RESET}' if result==3 else f'{RED}MISSING_IMAGE_FILE{RESET}' if result==4 else f'{GREEN}EXCLUDED_CLASS{RESET}' if result==5 else f'{RED}ERROR{RESET}'
                  print ( f"\033[{start_row+my_thread};{start_column+210}f{RESET}{r} \033[{start_row+my_thread};{start_column+227}f{RESET}tiles={MIKADO}{cumulative_tiles_processed:,}{RESET}", flush=True  )                
  
      if slides_processed>=my_expanded_slide_quota:
        break

    if slides_processed>=my_expanded_slide_quota:
      break

  if DEBUG>0:
    if slides_processed==my_slide_quota:
      print ( f"\033[{start_row+my_thread};{start_column+130}f       {RESET}{GREEN  }thread {MIKADO}{my_thread:2d}{RESET}{GREEN  } exiting - on    slide quota{RESET}", flush=True  )
    elif slides_processed>my_slide_quota:
      print ( f"\033[{start_row+my_thread};{start_column+130}f       {RESET}{MAGENTA}thread {MIKADO}{my_thread:2d}{RESET}{MAGENTA} exiting - over  slide quota{RESET}", flush=True )
    else:
      print ( f"\033[{start_row+my_thread};{start_column+130}f       {RESET}{RED    }thread {MIKADO}{my_thread:2d}{RESET}{RED    } exiting - under slide quota{RESET}", flush=True )

  return( slides_processed )
