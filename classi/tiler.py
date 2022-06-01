"""

This routine performs tiling for exactly one SVS image

"""

import os
import re
import sys
import cv2
import time
import math
import datetime
import glob
import random
import argparse
import multiprocessing
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**32)
import openslide
import numpy as np

# ~ import tkinter as tk
# ~ from   tkinter  import Label, Tk, Canvas

from random             import randint
from norms              import Normalizer
# ~ from PIL                import ImageTk
from PIL                import Image
from PIL                import ImageFont
from PIL                import ImageDraw
from shutil             import copyfile as cp
from scipy.stats.mstats import ttest_1samp
from torchvision        import transforms

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=350)

BB="\033[35;1m"

from constants  import *



DEBUG=1

num_cpus = multiprocessing.cpu_count()

start_column = 170
start_row    = 60-num_cpus

thread_to_monitor = 7

def tiler( args, r_norm, n_tiles, tile_size, batch_size, stain_norm, norm_method, d, f, my_thread ):

  num_cpus = multiprocessing.cpu_count()


  SUCCESS=True
  FAIL=False
  
  a = random.choice( range(150+2*my_thread,255) )
  b = random.choice( range(50,225) )
  c = random.choice( range(50,225) )
  BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

 
  tiles_processed                 = 0
  tiles_considered_count          = 0
  background_image_count          = 0
  low_contrast_tile_count         = 0
  degenerate_image_count          = 0
  background_image_count          = 0
  stain_normalization_target_set  = False
  
  # DON'T USE args.n_tiles or args.tile_size, since it is the JOB level list of numbers of tiles. Here we are just using one value of n_tiles, passed in as the parameter above  
  just_profile           = args.just_profile                                                               # display an analysis of image tiles then exit
  just_test              = args.just_test                                                                  # if set, suppress tile quality filters (i.e. accept every tile)
  multimode              = args.multimode                                                                  # if set, suppress tile quality filters (i.e. accept every tile)
  data_dir               = args.data_dir
  log_dir                = args.log_dir
  rand_tiles             = args.rand_tiles                                                                 # select tiles at random coordinates from image. Done AFTER other quality filtering
  zoom_out_prob          = args.zoom_out_prob
  zoom_out_mags          = args.zoom_out_mags
  greyness               = args.greyness                                                                   # Used to filter out images with very low information value
  min_uniques            = args.min_uniques                                                                # tile must have at least this many unique values or it will be assumed to be degenerate
  min_tile_sd            = args.min_tile_sd                                                                # Used to cull slides with a very reduced greyscale palette such as background tiles 
  points_to_sample       = args.points_to_sample                                                           # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination
  supergrid_size         = args.supergrid_size
  scattergram            = args.scattergram
    
  if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):  
    greyness=60
    min_uniques=100
    min_tile_sd=3
    points_to_sample=100
    

  if not just_profile=='True':
    if (DEBUG>2):
      if ( ( just_test!='True' ) | ( multimode=='image_rna') ):      
        print ( f"process:slide: {BB}{my_thread}) {f:66s}{RESET} ", flush=True, end="" )
      else:
        print ( f"TILER:            INFO: process:slide                 = {CYAN}{my_thread:2d}{RESET}:{f:66s} ", flush=True         )
        
  already_displayed=False
      
  if (DEBUG>9):
    print ( f"TILER:          INFO: (parent directory)  = {BB}{f}{RESET}",                  flush=True)
    print ( f"TILER:          INFO: (thread num)        = {BB}{my_thread}{RESET}",          flush=True)
    print ( f"TILER:          INFO: (stain_norm)        = {BB}{stain_norm}{RESET}",         flush=True)
    #print ( f"TILER: INFO: (thread num)        = {BB}{stain_norm_target}{RESET}",  flush=True)

  fqn = f"{data_dir}/{d}/{f}"
  
  if (DEBUG>2):
    print ( f"TILER:          INFO: process:directory = {CYAN}{my_thread:2d}{RESET}:{d:100s} ", flush=True         )
  
  ALLOW_REDUCED_WIDTH_EDGE_TILES = 0                                                                       # if 1, then smaller tiles will be generated, if required, at the right hand edge and bottom of the image to ensure that all of the image is tiled
  
  level            = 0
  tile_size_40X    = 2100;                                                                                 # only used if resizing is enabled (tile_size=0)
  
  start = time.time()
  
  if (DEBUG>2):  
    print('TILER:          INFO: now processing          {:}{:}{:}'.format( BB, fqn, RESET));
  
  # (1) open the SVS image and inspect statistics
  
  try:
    oslide = openslide.OpenSlide( fqn );                                                                   # open the file containing the image

    if (tile_size==0):                                                                                     # PGD 191217
      tile_width = int(tile_size_40X * mag / 40)                                                           # scale tile size from 40X to 'mag'. 'tile_size_40X' is set above to be 2100
    else:                                                                                                  # PGD 191217
      tile_width = tile_size                                                                               # PGD 191231
      
    width  = oslide.dimensions[0];                                                                         # width  of slide image
    height = oslide.dimensions[1];                                                                         # height of slide image

  except Exception as e:
    print( f"{RED}TILER_{my_thread}:                   ERROR: exception occured in tiler thread {MIKADO}{my_thread}{RESET}{RED} !!! {RESET}"        )
    print( f"{RED}TILER_{my_thread}:                          exception text: '{CYAN}{e}{RESET}{RED} ... halting now"  )
    sys.exit(0);

  potential_tiles = (width-tile_width)*(height-tile_width) // (tile_width*tile_width)
  if (DEBUG>1):
    if not just_profile=='True':
      print( f"TILER:          INFO: slide height x width (pixels) = {BB}{height:6d} x {width:6d}{RESET} and potential ({BB}{tile_width:3d}x{tile_width:3d}{RESET} sized ) tiles for this slide = {BB}{potential_tiles:7d}{RESET} ", end ="", flush=True )

  if potential_tiles<n_tiles:
    print( f"\n{ORANGE}TILER:          WARNING: requested tiles (n_tiles) = {CYAN}{n_tiles:,}{RESET}{ORANGE} but only {RESET}{CYAN}{potential_tiles:,}{RESET}{ORANGE} possible. Slide will be skipped. ({CYAN}{fqn}{RESET}{ORANGE}){RESET}", flush=True)
    return FAIL


  if openslide.PROPERTY_NAME_VENDOR in oslide.properties:
    PROPERTY_NAME_VENDOR = oslide.properties[ openslide.PROPERTY_NAME_VENDOR]
    if (DEBUG>99):
      print(f"\033[{start_row+num_cpus+1};0f\r{CLEAR_LINE}{DULL_WHITE}TILER:          INFO: PROPERTY_NAME_VENDOR          = {BB}{PROPERTY_NAME_VENDOR}{RESET}", flush=True )
  
  if just_test != 'True':
    if openslide.PROPERTY_NAME_COMMENT in oslide.properties:
      PROPERTY_NAME_COMMENT = oslide.properties[ openslide.PROPERTY_NAME_COMMENT]                          
      PROPERTY_NAME_COMMENT = re.sub( r'\n' , ' ', PROPERTY_NAME_COMMENT )                                   # PROPERTY_NAME_COMMENT has an embedded carriage return and line feed so strip these out
      PROPERTY_NAME_COMMENT = re.sub( r'\r' , ' ', PROPERTY_NAME_COMMENT )  
      if (DEBUG>0):
        print(f"{SAVE_CURSOR}", end="",                                flush=True )
        print(f"\033[{start_row+num_cpus};0f\r{CLEAR_LINE}", end="", flush=True )
        print(f"\033[{start_row+num_cpus+1};0f\r{CLEAR_LINE}", end="", flush=True )
        print(f"\033[{start_row+num_cpus+3};0f\r{CLEAR_LINE}", end="", flush=True )
        print(f"\033[{start_row+num_cpus+4};0f\r{CLEAR_LINE}", end="", flush=True )
        print(f"\033[{start_row+num_cpus+2};0f\r{CLEAR_LINE}{BB}{PROPERTY_NAME_COMMENT[0:300]}{RESET}{RESTORE_CURSOR}", flush=True )
      
  objective_power = 0
  if openslide.PROPERTY_NAME_OBJECTIVE_POWER in oslide.properties:
    objective_power = int(oslide.properties[ openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    if (DEBUG>2):
      print(f"\r{DULL_WHITE}TILER:          INFO: objective power         = {BB}{objective_power}{RESET}", flush=True )
  else:
    if (DEBUG>2):
      print(f"\r{DULL_WHITE}TILER:          INFO: objective power         = {DULL_WHITE}property {BB}PROPERTY_NAME_OBJECTIVE_POWER{RESET}{DULL_WHITE} does not exist for this slide{RESET}")
             
  if openslide.PROPERTY_NAME_MPP_X           in oslide.properties:                                         # microns per pixel the image was scanned at
    if (DEBUG>2):
      print(f"\r{DULL_WHITE}                         microns/pixel = {BB}PROPERTY_NAME_MPP_X{RESET} = {MIKADO}{float(oslide.properties[ openslide.PROPERTY_NAME_MPP_X]):6.2f}{RESET}", flush=True )                
  elif "XResolution" in oslide.properties:                                                                 # for TIFF format images (apparently)  https://openslide.org/docs/properties/
    mag = 10.0 / float(oslide.properties["XResolution"]);
    if (DEBUG>2):
      print(f"\r{DULL_WHITE}TILER:          INFO:   XResolution       = {DULL_WHITE}property {BB}XResolution{RESET} = {MIKADO}{float(oslide.properties['XResolution']):6.2f}{RESET}",                              flush=True )
      print(f"\r{DULL_WHITE}TILER:          INFO:   magnification                                                               = {MIKADO}float(oslide.properties['XResolution'] / {10.0} = {mag:6.2f}{RESET}",  flush=True ) 
  else:
    if (DEBUG>2):
      print(f"\r{DULL_WHITE}TILER:          INFO:   Neither {CAMEL}PROPERTY_NAME_MPP_X{RESET} nor {BB}XResolution{RESET} exist for this slide{RESET}")



  """
  if not stain_norm =="NONE":                                                                  # then perform the selected stain normalization technique on the tile

      # First way is to provide Normalizer with mean and std parameters
      # Mean(r, g, b) = (0, 0, 0), Std(r, g, b) = (1, 1, 1) 
      # normalization_target = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

      # Second way is to provide Normalizer with a target image. It will normalize all other tiles to match the target image

      normalization_target = fqn
    
      if (DEBUG>9):
        print ( f"TILER:     INFO:  about to call 'Normalizer' with parameters \033[35m{stain_norm}\033[m and 'normalization_parameters' matrix", flush=True ) 
    
      norm_method = Normalizer( stain_norm, normalization_target )                           #  one of <reinhard, spcn>;  target: Path of target image to normalize images to OR normalization_parameters as per above
    
      if (DEBUG>9):
        print ( f"TILER:     INFO:  norm_method.method = \033[36m{norm_method.method}\033[m,  norm_method.normalizer = \033[36m{norm_method.normalizer}\033[m",   flush=True )
   """



  # (2a) [test mode] look for the best possible patch (of the requested size) to use
     
  if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):

    samples      = args.patch_points_to_sample
    high_uniques = 0
    if DEBUG>2:
      print( f"\r{WHITE}TILER:          INFO: about to analyse {MIKADO}{samples}{RESET} randomly selected {MIKADO}{int(n_tiles**0.5)}x{int(n_tiles**0.5)}{RESET} patches to locate a patch with high nominal contrast and little background{RESET}" )  
    x_start, y_start, high_uniques = highest_uniques( args, oslide, level, width, height, tile_width, samples, n_tiles )
    if high_uniques==0:                                                                                    # means we went found no qualifying tile to define the patch by (can happen)
      x_start=int( width//2)
      y_start=int(height//2)
      print( f"\033[38;2;255;165;0m\033[1mTILER:            INFO:  no suitable patch found: setting coordinates to centre of slide x={x_start:7d} y={y_start:7d}\033[m" )
    else:
      if DEBUG>1:
        print( f"\033[1m\033[mTILER:            INFO:  coordinates of tile in slide with best contrast: x={x_start:7d} y={y_start:7d} and highest number of unique RGB values = {high_uniques:5d}\033[m" )

    if DEBUG>2:
      print( f"{ORANGE}TILER:          INFO: CAUTION! 'just_test' flag is set (and multimode flag is not). (Super-)patch origin will be set to the following coordinates, chosen for good contrast: x={CYAN}{x_start}{RESET}{ORANGE}, y={CYAN}{y_start}{RESET}" )  
  
  
  # (2b) Set up parameters for selection of tiles (for training mode and multimode: random; for test mode: 2D contiguous patch taking into account the supergrid setting
  
  if ( ( just_test!='True' ) | ( multimode=='image_rna' ) ):
    x_start=0
    y_start=0
    x_span=range(x_start, width, tile_width)                                                               # steps of tile_width
    y_span=range(y_start, width, tile_width)                                                               # steps of tile_width
    

  else:                                                                                                    # test mode (for patching)
    tiles_to_get = int(batch_size**0.5)                                                                    # length of one side of the patch, in number of tiles (the patch is square, and the  batch_size is chosen to be precisely equal to the n_tiles for test mode) 
    tile_height  = tile_width
    patch_width  = (tiles_to_get*supergrid_size*tile_width)                                                # multiply by tile_width to get pixels
    patch_height = (tiles_to_get*supergrid_size*tile_width)                                                # multiply by tile_width to get pixels
    x_span=range(x_start, x_start + (tiles_to_get*supergrid_size*tile_width), tile_width)                  # steps of tile_width
    y_span=range(y_start, y_start + (tiles_to_get*supergrid_size*tile_width), tile_height)                 # steps of tile_height
    
    
  if DEBUG>18:
    if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):  
      supergrid_side = int(supergrid_size*batch_size**0.5)
      print( f"{WHITE}TILER:          INFO:    supergrid       (user parameter) = {MIKADO}{supergrid_size}{RESET}" )  
      print( f"{WHITE}TILER:          INFO:    tiles per batch (user parameter) = {MIKADO}{batch_size}{RESET}" )
      print( f"{WHITE}TILER:          INFO:      hence supergrid dimensions                     = {MIKADO}{supergrid_size}x{supergrid_size}{RESET}" )
      print( f"{WHITE}TILER:          INFO:      hence supergrid height x width                 = {MIKADO}{supergrid_side}x{supergrid_side}{WHITE}        tiles{RESET}" )
      print( f"{WHITE}TILER:          INFO:      hence supergrid height x width                 = {MIKADO}{patch_width:,}x{patch_width:,}{WHITE}  pixels{RESET}" )
      print( f"{WHITE}TILER:          INFO:      hence supergrid size                           = {MIKADO}{patch_width*patch_width/1000000:.1f}{WHITE}          Megapixels{RESET}" )
      print( f"{WHITE}TILER:          INFO:      hence supergrid total tiles                    = {MIKADO}{batch_size*supergrid_size**2:,} {RESET}" ) 
      print( f"{WHITE}TILER:          INFO:      hence number of batches required for supergrid = {MIKADO}{supergrid_size**2}{RESET}" )      
    if DEBUG>99:                 
      print( f"{WHITE}TILER:          INFO:  x_span (pixels)               = {x_span}{RESET}" )
      print( f"{WHITE}TILER:          INFO:  y_span (pixels)               = {y_span}{RESET}" )
      print( f"{WHITE}TILER:          INFO:  x_start (pixel coords)        = {x_start}{RESET}" )
      print( f"{WHITE}TILER:          INFO:  y_start (pixel coords)        = {y_start}{RESET}" ) 



  # (2c) [test mode] extract and save a copy of the entire un-tiled patch, for later use in the Tensorboard scattergram display
  
  if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):  
            
    patch       = oslide.read_region((x_start, y_start), level, (patch_width, patch_height))               # matplotlibs' native format is PIL RGBA
    
    patch_rgb   = patch.convert('RGB')                                                                     # convert from PIL RGBA to RGB
    patch_npy   = (np.array(patch))                                                                        # convert to Numpy array
    patch_fname = f"{data_dir}/{d}/entire_patch.npy"                                                       # same name for all patch since they are in different subdirectories of data_dur
    #fname = '{0:}/{1:}/{2:06}_{3:06}.png'.format( data_dir, d, x_rand, y_rand)
    np.save(patch_fname, patch_npy)
      
    if (DEBUG>2):
      print ( f"{CLEAR_LINE}TILER:          INFO:      patch_fname                                    = {MAGENTA}{patch_fname}{RESET}" )
      
 # patch = patch_norm_PIL.convert("RGB")
 # patch_norm_PIL = Image.fromarray( patch_uint8 )
 # patch_uint8 = np.uint8( patch_255 )
 # patch_255 = patch_norm * 255
 # patch_norm = norm_method.normalizer( patch_rgb_npy )
  
  
  # (3) extract the tiles

  
  break_now=False


  for x in x_span:

      if break_now==True:
        break

      for y in y_span:
    
          tiles_considered_count+=1
              
          if   ( ( just_test=='True' ) & ( multimode!='image_rna' ) ) & ( tiles_processed==n_tiles*(supergrid_size**2) ):
            break_now=True
            break
          elif ( ( just_test!='True' ) | ( multimode=='image_rna') )  & ( tiles_processed==n_tiles  ):
            break_now=True
            break            
              
          else:
            if (x>width-2*tile_width) & (y>height-2*tile_width):
              if just_profile=='True':
                if already_displayed==False:
                  print(f'\n{ORANGE}TILER: WARNING: not enough tiles in this slide to meet the quality criteria. (At coords {CYAN}{x},{y}{RESET}) with {CYAN}{tiles_processed}{RESET}) -- skipping {CYAN}{fqn}{RESET}', flush=True)
                  already_displayed=True
              else:
                if ( ( just_test!='True' ) | ( multimode=='image_rna') ):
                  print(f'\n{ORANGE}TILER: WARNING: not enough tiles in slide to meet the quality criteria. (At coords {CYAN}{x},{y}{RESET}) with {CYAN}{tiles_processed}{RESET}) -- skipping {CYAN}{fqn}{RESET}', flush=True)
                  return FAIL
              
            if x + tile_width > width:
                pass
            else:
                tile_width_x = tile_width;
            if y + tile_width > height:
                pass
            else:
                tile_width_y = tile_width
                        
            x_resize = int(np.ceil(tile_size_40X * tile_width_x/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
            y_resize = int(np.ceil(tile_size_40X * tile_width_y/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
 
            x_rand = randint( 1, (width  - tile_width_x)) 
            y_rand = randint( 1, (height - tile_width_y)) 
            

            if zoom_out_prob[0] != 1:
              multiplier = choose_mag_level( my_thread, zoom_out_prob, zoom_out_mags, r_norm )
              if DEBUG>3:
                print ( f"{RESET}TILER:          INFO: multiplier  = {MIKADO}{multiplier}{RESET}" )
            else:
              multiplier = 1

            if DEBUG>0:
              print( f"\r\033[{start_row-1}{CLEAR_LINE}" )

            if DEBUG>0:
              if objective_power==0:
                print( f"\r\033[{start_row+my_thread};0H{RESET}       objective_power = {ARYLIDE}{objective_power:2d}{RESET} (means it was not recorded. will assume 40x)                 ", end="", flush=True )
              elif objective_power==20:
                print( f"\r\033[{start_row+my_thread};0H{RESET}       objective_power = {ARYLIDE}{objective_power:2d}{RESET} (will extract double tile size then shrink)                  ", end="", flush=True )
              else:
                print( f"\r\033[{start_row+my_thread};0H{RESET}       objective_power = {CAMEL}{objective_power:2d}{RESET}                                                              ",   end="", flush=True )                
              print( f"\r\033[{start_row+my_thread};90H{RESET}zoom out = \
{AMETHYST if multiplier==1 else MIKADO if multiplier==2 else CARRIBEAN_GREEN if 2<multiplier<=4 else BITTER_SWEET if 5<multiplier<=8 else CHARTREUSE if 5<multiplier<=8 else CAMEL}{multiplier}{RESET}               ", end="", flush=True )

            if DEBUG>0:
                print  (f"\
      {WHITE}\
\r\033[{start_row-2};{start_column+3}fcpu\
\r\033[{start_row-2};{start_column+15}fexamined\
\r\033[{start_row-2};{start_column+33}faccepted\
\r\033[{start_row-2};{start_column+48}flow_contrast\
\r\033[{start_row-2};{start_column+64}fdegenerate\
\r\033[{start_row-2};{start_column+80}fbackground\
", flush=True, end="" )

            if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):
              
              if objective_power==20:

                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print( f"{CARRIBEAN_GREEN}TILER_{my_thread}:        INFO:  objective_power               = {GREEN}{objective_power}{RESET}     << Need to increase resolution of this tile",  flush=True  )
                new_width = multiplier*2*tile_width_x
                tile = oslide.read_region((x     ,  y     ),  level, (new_width, new_width))               # extract an area from the slide of size determined by the result returned by choose_mag_level
                if (DEBUG>0):
                  print ( f"\r\033[{start_row+my_thread};{start_column-48}f\033[{7*int(math.log2(multiplier))}C{CARRIBEAN_GREEN}{new_width}x{new_width}{RESET}                                          " )
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{GREEN}{np.array(tile)[0:20,0:20,0]}{RESET}",  flush=True        ) 
                tile = tile.resize((tile_width_x,tile_width_x),Image.ANTIALIAS)                            # shrink it to tile_size
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{GOLD}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True         )
                  
              else:

                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print( f"{CARRIBEAN_GREEN}TILER_{my_thread}:        INFO:  objective_power               = {GREEN}{objective_power}{RESET}     << NO need to increase resolution of this tile",  flush=True  )
                new_width = multiplier*1*tile_width_x
                tile = oslide.read_region((x     ,  y     ),  level, (new_width, new_width))               # extract an area from the slide of size determined by the result returned by choose_mag_level
                if (DEBUG>0):
                  print ( f"\r\033[{start_row+my_thread};{start_column-48}f\033[{7*int(math.log2(multiplier))}C{GREEN}{new_width}x{new_width}{RESET}                                          " )
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{BITTER_SWEET}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True        ) 
                tile = tile.resize((tile_width_x,tile_width_x),Image.ANTIALIAS)                            # shrink it to tile_size
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{BITTER_SWEET}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True    )           

              fname = '{0:}/{1:}/{2:06}_{3:06}.png'.format( data_dir, d, y, x)                             # use the tile's top-left coordinate to construct a unique filename
     
                 

            else:
              
              if objective_power==20:
                    
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print( f"{CARRIBEAN_GREEN}TILER_{my_thread}:        INFO:  objective_power               = {GREEN}{objective_power}{RESET}     << Need to increase resolution of this tile",  flush=True  )
                new_width = multiplier*2*tile_width_x
                tile = oslide.read_region((x_rand,  y_rand),  level, (new_width, new_width))               # extract an area from the slide of size determined by the result returned by choose_mag_level
                if (DEBUG>0):
                  print ( f"\r\033[{start_row+my_thread};{start_column-48}f\033[{7*int(math.log2(multiplier))}C{CARRIBEAN_GREEN}{new_width}x{new_width}{RESET}                                          " )
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{GREEN}{np.array(tile)[0:20,0:20,0]}{RESET}",  flush=True        ) 
                tile = tile.resize((tile_width_x,tile_width_x),Image.ANTIALIAS)                            # shrink it to tile_size
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{GOLD}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True         )

                if (DEBUG>99):
                  time.sleep(0.1)
                

              else:
                
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print( f"{CARRIBEAN_GREEN}TILER_{my_thread}:        INFO:  objective_power               = {GREEN}{objective_power}{RESET}     << NO need to increase resolution of this tile",  flush=True  )
                new_width = multiplier*1*tile_width_x
                tile = oslide.read_region((x_rand,  y_rand),  level, (new_width, new_width))               # extract an area from the slide of size determined by the result returned by choose_mag_level
                if (DEBUG>0):
                  print ( f"\r\033[{start_row+my_thread};{start_column-48}f\033[{7*int(math.log2(multiplier))}C{BITTER_SWEET}{new_width}x{new_width}{RESET}                                          " )
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{CARRIBEAN_GREEN}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True        ) 
                tile = tile.resize((tile_width_x,tile_width_x),Image.ANTIALIAS)                            # shrink it to tile_size
                if (DEBUG>5) & (my_thread==thread_to_monitor):
                  print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile (PIL RGBA) after resizing = \n{BITTER_SWEET}{np.array(tile)[0:10,0:10,0]}{RESET}",  flush=True    )

                if (DEBUG>99):
                  time.sleep(0.1)


              fname = f'{data_dir:}/{d:}/{x_rand:06}_{y_rand:06}_{objective_power}_{new_width}x{new_width}.png'                    # use the tile's top-left coordinate to construct a unique filename



            if DEBUG>9:
              print ( f"{RESET}TILER_{my_thread}:          INFO: shape (tile as numpy array)  = {CYAN}{(np.array(tile)).shape}                    {RESET}" )
              print ( f"{RESET}TILER_{my_thread}:          INFO:                  type(tile)  = {CYAN}{type(tile)}{RESET}" ) 
            if (DEBUG>999):
              print ( f"{RESET}TILER_{my_thread}:          INFO: \r\033[25Ctile -> numpy array = {YELLOW}{np.array(tile)[0:10,0,0]}{RESET}\r\033[90Ctile -> RGB -> numpy array = {BLEU}{np.array(tile.convert('RGB'))[0:10,0,0]}                   {RESET}",                 flush=True    ) 


            if (DEBUG>999):
              print ( f"{MAGENT}TILER:         CAUTION:                                 about to emboss tile with file name for debugging purposes{RESET}" )
              tile_dir=f"{d[-6:]}"
              x_coord=f"{x}"
              y_coord=f"{y}"                
              draw = ImageDraw.Draw(tile)
              font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 75)
              draw.text( (5,  20),  tile_dir  ,(0,0,0), font=font )
              draw.text( (5, 100),  y_coord   ,(0,0,0), font=font )                     
              draw.text( (5, 180),  x_coord   ,(0,0,0), font=font )


            if (tile_size==0):                                                                             # tile_size=0 means resizing is desired by user
              tile = tile.resize((x_resize, y_resize), Image.ANTIALIAS)                                    # resize the tile; use anti-aliasing option


            # decide by means of a heuristic whether the tile contains is background or else contains too much background
            IsBackground   = check_background( args, tile )
            if IsBackground:
              background_image_count+=1
            
            # decide by means of a heuristic whether the tile is of low contrast
            IsLowContrast = check_contrast   ( args, tile )
            if IsLowContrast:
              low_contrast_tile_count       +=1

            # check the number of unique values in the image, tile we will use as a proxy to discover degenerate (images)
            IsDegenerate  = check_degeneracy ( args, tile )
            if IsDegenerate:
              degenerate_image_count+=1

            if ( IsBackground | IsDegenerate | IsLowContrast ) & ( ( just_test!='True' ) | ( multimode=='True') ):                   # If 'just_test' = True, all tiles must be accepted
              if (DEBUG>999):
                print ( "TILER: INFO:               skipping this tile" ) 
              pass
      

            else:
              # ~ if not stain_norm=="NONE":                                                               # then perform the selected stain normalization technique on the tile W
              if stain_norm=="reinhard":                                                                   # now handle 'spcn' at the slide level in the standalone process 'normalise_stain' 


                """
                if stain_normalization_target_set==False:                                                  # do one time per slide only
                  stain_normalization_target_set=True

                  # First way is to provide Normalizer with mean and std parameters
                  # Mean(r, g, b) = (0, 0, 0), Std(r, g, b) = (1, 1, 1) 
                  # normalization_target = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

                  # Second way is to provide Normalizer with a target image. It will normalize all other tiles to match the target image
                  tile_rgb     = tile.convert('RGB')
                  tile_rgb_npy = (np.array(tile_rgb))
                  normalization_target = tile_rgb_npy
                
                  if (DEBUG>9):
                    print ( f"TILER:     INFO:  about to call 'Normalizer' with parameters \033[35m{stain_norm}\033[m and 'normalization_parameters' matrix", flush=True ) 
                
                  norm_method = Normalizer( stain_norm, normalization_target )                             #  one of <reinhard, spcn>;  target: Path of target image to normalize images to OR normalization_parameters as per above
                
                  if (DEBUG>9):
                    print ( f"TILER:     INFO:  norm_method.method = \033[36m{norm_method.method}\033[m,  norm_method.normalizer = \033[36m{norm_method.normalizer}\033[m",   flush=True )
                 """

                tile = stain_normalization( norm_method, tile  )                                           # returns stain normalized version of the tile
              

              if (DEBUG>9):
                print ( "TILER: INFO:               x = \033[1m{:}\033[m".format(x),             flush=True)
                print ( "TILER: INFO:               y = \033[1m{:}\033[m".format(y),             flush=True)  
                print ( "TILER: INFO:      tile_width = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:     tile_height = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:          fname  = \033[1m{:}\033[m".format( fname ) )

              if (DEBUG>999):
                print ( f"{RESET}\rTILER: INFO: \r\033[25Ctile -> numpy array = {YELLOW}{np.array(tile)[0:10,0,0]}{RESET}",                 flush=True    ) 

              tile.save(fname);                                                                            # save to the filename we made for this tile earlier              
              tiles_processed += 1

              # "Note that just calling a file .png doesn't make it one so you need to specify the file format as a second parameter.  tile.save("tile.png","PNG")"
              
#             print ( "\033[s\033[{:};{:}f\033[32;1m{:}{:2d};{:>4d} \033[m\033[u".format( randint(1,68), int(1500/num_cpus)+7*my_thread, BB, my_thread+1, tiles_processed ), end="", flush=True )
              if (DEBUG>99):
                print ( f"{SAVE_CURSOR}\033[{tiles_processed//50};{int(1500/num_cpus)+7*my_thread}f\033[32;1m{BB}{my_thread+1:2d};{tiles_processed:>4d} {RESET}{RESTORE_CURSOR}", end="", flush=True )
    
          if just_profile=='True':
            print ( f"\
         \033[34mslide=\033[1m{f:66s}{RESET}\
         \033[34mheightxwidth=\033[1m{height:6d} x{height:6d}{RESET}\
         \033[34mavailable {tile_width_x:3d} x{tile_width_x:3d} tiles=\033[1m{potential_tiles:6d} {RESET}\
         \033[34manalysed=\033[1m{tiles_considered_count:6d} {RESET}\
         \033[34macceptable=\033[1m{tiles_processed:5d} \
         \033[1m({tiles_processed/tiles_considered_count *100:2.0f})%{RESET}\
         \033[34mlow contrast=\033[1m{low_contrast_tile_count:5d};\
         \033[1m({low_contrast_tile_count/tiles_considered_count *100:2.1f})%{RESET}\
         \033[34mdegenerate=\033[1m{degenerate_image_count:5d} \
         \033[1m({degenerate_image_count/tiles_considered_count *100:2.1f}%){RESET}\
         \033[34mbackground=\033[1m{background_image_count:5d} \
         \033[1m({background_image_count/tiles_considered_count *100:2.0f})% {RESET}", flush=True )

          else:
            if (DEBUG>0):
              pass
              if ( ( just_test!='True' ) | ( multimode=='True') ):
                pass
                # ~ time.sleep(0.2)
                # ~ print ( f"{SAVE_CURSOR}\033[{my_thread+67-num_cpus};{start_column}f", end="" )
              else:
                pass
                # ~ print ( f"{SAVE_CURSOR}{CLEAR_LINE}", end="" )

              print  (f"\
{RESET}{BRIGHT_GREEN if tiles_processed>=(0.95*n_tiles) else GREEN if tiles_processed>=(0.90*n_tiles) else ORANGE if tiles_processed>=(0.75*n_tiles) else ORANGE if tiles_processed>=(0.25*n_tiles) else WHITE if tiles_processed<=(0.10*n_tiles) else BLEU}\
\r\033[{start_row+my_thread};{start_column}f{my_thread:^8d}      \
\r\033[{start_row+my_thread};{start_column+12}f{tiles_considered_count:6d}  \
  ({(tiles_processed/n_tiles*100):>3.0f}%)   \
\r\033[{start_row+my_thread};{start_column+30}f{tiles_processed:6d}  ({tiles_processed/[tiles_considered_count                 if tiles_considered_count>0 else .000000001][0] *100:4.0f}%)  \
\r\033[{start_row+my_thread};{start_column+46}f{low_contrast_tile_count:6d}  ({low_contrast_tile_count/[tiles_considered_count if tiles_considered_count>0 else .000000001][0] *100:4.0f}%)  \
\r\033[{start_row+my_thread};{start_column+62}f{degenerate_image_count:6d}  ({degenerate_image_count/[tiles_considered_count   if tiles_considered_count>0 else .000000001][0] *100:4.0f}%)  \
\r\033[{start_row+my_thread};{start_column+78}f{background_image_count:6d}  ({background_image_count/[tiles_considered_count   if tiles_considered_count>0 else .000000001][0] *100:4.0f}%)  \
", flush=True, end="" )

              # ~ time.sleep(.25)
              print ( f"{RESTORE_CURSOR}", end="" )
  
  if (DEBUG>9):
    print('TILER: INFO: time taken to tile this SVS image: \033[1m{0:.2f}s\033[m'.format((time.time() - start)/60.0))

  fq_name = f"{data_dir}/{d}/SLIDE_TILED"

  with open(fq_name, 'w') as f:
    f.write( f"this SVS or TIF file has been tiled" )
  f.close  
  

  # ~ if (DEBUG>9):
    # ~ print ( "TILER: INFO: about to display the \033[33;1m{:,}\033[m tiles".format    ( tiles_processed   ) )
    # ~ result = display_processed_tiles( data_dir, DEBUG )

    
  return SUCCESS

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def button_click_exit_mainloop (event):
    event.widget.quit()                                                                                    # this will cause main loop to unblock.

# ------------------------------------------------------------------------------

# ~ def display_processed_tiles( the_dir, DEBUG ):

# ~ # from: https://code.activestate.com/recipes/521918-pil-and-tkinter-to-display-images/

  # ~ if (DEBUG>9):
    # ~ print ( "TILER: INFO: at top of display_processed_tiles() and dir = \033[33;1m{:}\033[m".format( the_dir   ) )

  # ~ dirlist         = os.listdir( the_dir )

  # ~ for f in dirlist:
    # ~ if (DEBUG>9):
      # ~ print ( "TILER: INFO: display_processed_tiles() current file      = \033[33;1m{:}\033[m".format( f  ) )
      # ~ try:
          # ~ master = Tk()
          # ~ master.bind("<Button>", button_click_exit_mainloop )
          # ~ master.geometry('+%d+%d' % (1350,500))                                                           # set window position
          # ~ old_label_image = None
          # ~ image1 = Image.open(f)                                                                           # open the file
          # ~ resized = image1.resize((512, 512),Image.ANTIALIAS)
          # ~ master.geometry('%dx%d' % (resized.size[0],resized.size[1]))                                     # set the size to be the same dimensions as a tile
          # ~ tkpi = ImageTk.PhotoImage(resized)                                                               # convert the png image into a canonical tkinter image object (tkinter doesn't natively support png)
          # ~ label_image = tk.Label(master, image=tkpi)                                                       # create a tkinter 'Label' display object
          # ~ label_image.tkpi=tkpi
          # ~ label_image.pack()                                                                               # 
          # ~ #label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])                           #
          # ~ master.title(f)                                                                                  # use the file name as the image title
          # ~ if old_label_image is not None:
              # ~ old_label_image.destroy()
          # ~ old_label_image = label_image
          # ~ master.mainloop()                                                                                # wait for user input (which we simulate with button_click_exit_mainloop())
      # ~ except Exception as e:
          # ~ if (DEBUG>9):
            # ~ print ( "TILER: INFO: Exception                                   = {:}".format( e  ) )
          # ~ # skip anything not an image
          # ~ # Warning, this will hide other errors as well
          # ~ pass

  # ~ return SUCCESS
                  

# ------------------------------------------------------------------------------

def stain_normalization( norm_method, tile ):
  
  tile_rgb     = tile.convert('RGB')
  tile_rgb_npy = (np.array(tile_rgb))

  if (DEBUG>9):
    print ( "TILER:            INFO: performing \033[35m{:}\033[m stain normalization on tile \033[35m{:}\033[m".format    ( stain_norm, fname  ), flush=True )

  if (DEBUG>1):
    print ( "TILER:     INFO:  norm_method.normalizer              = \033[36m{:}\033[m".format( norm_method.normalizer), flush=True )

  tile_norm = norm_method.normalizer( tile_rgb_npy )                  #  ( path of source image )
  
  if (DEBUG>1):
    print ( "TILER:            INFO: norm_method.normalizer              = \033[36m{:}\033[m".format( norm_method.normalizer), flush=True )
    print ( "TILER:            INFO: shape of stain normalized tile      = \033[36m{:}\033[m".format( tile_norm.shape ), flush=True )
  if (DEBUG>99):
    print ( "TILER:            INFO: stain normalized tile               = \033[36m{:}\033[m".format( tile_norm       ), flush=True )

  tile_255 = tile_norm * 255
  if (DEBUG>99):
    np.set_printoptions(formatter={'float': lambda x: "{:3.2f}".format(x)})
    print ( "TILER:            INFO: stain normalized tile shifted to 0-255   = \033[36m{:}\033[m".format( tile_255       ), flush=True )  

  tile_uint8 = np.uint8( tile_255 )
  if (DEBUG>99):
    np.set_printoptions(formatter={'int': lambda x: "{:>3d}".format(x)})
    print ( "TILER:            INFO: stain normalized tile shifted to 0-255   = \033[36m{:}\033[m".format( tile_uint8       ), flush=True )   

  tile_norm_PIL = Image.fromarray( tile_uint8 )
  if (DEBUG>99):
    print ( "TILER:            INFO: stain normalized tile as RGP PIL   = \033[36m{:}\033[m".format( tile_norm_PIL       ), flush=True )
    
  #tile_norm_PIL = Image.fromarray( np.uint8( np.random.rand(128,128,3) * 255 ) ) 
  tile = tile_norm_PIL.convert("RGB")
  if (DEBUG>99):
    print ( "TILER:            INFO: stain normalized tile as RGP PIL   = \033[36m{:}\033[m".format( tile       ), flush=True )
    
  return tile

# ------------------------------------------------------------------------------
def check_background( args, tile ):

  tile_grey     = tile.convert('L')                                                                        # make a greyscale copy of the image
  np_tile_grey  = np.array(tile_grey)
  tile_PIL      = Image.fromarray( np_tile_grey )
  
  sample       = [  np_tile_grey[randint(0,np_tile_grey.shape[0]-1), randint(0,np_tile_grey.shape[0]-1)] for x in range(0, args.points_to_sample) ]
  sample_mean  = np.mean(sample)
  sample_sd    = np.std (sample)
  sample_t     = ttest_1samp(sample, popmean=sample_mean)

  IsBackground=False
  if sample_sd<args.min_tile_sd:
    IsBackground=True

    if (DEBUG>9):
      print ( "\nTILER:            INFO: highest_uniques(): sample \033[94m\n{:}\033[m)".format   (    sample     ) )
      print ( "TILER:            INFO: highest_uniques(): len(sample) \033[94;1m{:}\033[m".format ( len(sample)   ) )
      print ( "TILER:            INFO: highest_uniques(): sample_mean \033[94;1m{:}\033[m".format (  sample_mean  ) )          
      print ( "TILER:            INFO: highest_uniques(): sample_sd \033[94;1m{:}\033[m".format   (   sample_sd   ) )

    if (DEBUG>9):
      print ( f"TILER:            INFO: highest_uniques(): {RED}Yes, it's a background tile{RESET}",           flush=True )
  else:
    if (DEBUG>44):
      print ( f"TILER:            INFO: highest_uniques(): {BRIGHT_GREEN}No, it's not background tile{RESET}", flush=True )
      # ~ show_image ( tile_PIL )

  return IsBackground

# ------------------------------------------------------------------------------
def check_contrast( args, tile ):

  # check greyscale range, as a proxy for useful information content
  tile_grey         = tile.convert('L')                                                                    # make a greyscale copy of the image
  greyscale_range   = np.max(np.array(tile_grey)) - np.min(np.array(tile_grey))                            # calculate the range of the greyscale copy
  GreyscaleRangeOk  = greyscale_range>args.greyness
  GreyscaleRangeBad = not GreyscaleRangeOk

  if DEBUG>44:
    print ( f"TILER:            check_contrast()                   greyscale max   = {CAMEL}{  np.max(np.array(tile_grey)) }{RESET}"      )
    print ( f"TILER:            check_contrast()                   greyscale min   = {CAMEL}{  np.min(np.array(tile_grey)) }{RESET}"      )
    print ( f"TILER:            check_contrast()                                                                    greyscale range = { RED if GreyscaleRangeBad else BRIGHT_GREEN}{greyscale_range}{RESET}"  )
    time.sleep(1)
    
  if GreyscaleRangeBad:
    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a low contrast tile" )
      
  return GreyscaleRangeBad
  
# ------------------------------------------------------------------------------
def check_degeneracy( args, tile ):

  # check number of unique values in the image, which we will use as a proxy to discover degenerate (articial) images
  unique_values = len(np.unique(tile )) 
  if (DEBUG>4):
    print ( "TILER:            INFO: highest_uniques(): number of unique values in this tile = \033[94;1;4m{:>3}\033[m) and minimum required is \033[94;1;4m{:>3}\033[m)".format( unique_values, args.min_uniques ) )
  IsDegenerate = unique_values<args.min_uniques

  if DEBUG>44:
    print ( f"\n{RESET}{CLEAR_LINE}TILER:            check_degeneracy()  unique_values = {RED if IsDegenerate else BRIGHT_GREEN}{unique_values}{RESET}"      )
    time.sleep(1)
    
    
  if IsDegenerate:
    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a degenerate tile" )
      
  return IsDegenerate
  
# ------------------------------------------------------------------------------
def check_badness( args, tile ):

  IsDegenerate  = check_degeneracy (args, tile)
  IsLowContrast = check_contrast   (args, tile)
  IsBackground  = check_background (args, tile)
  
  IsBadTile = IsDegenerate | IsLowContrast | IsBackground
  
  if IsBadTile:
    if (DEBUG>999):
      print ( f"\033[1mTILER:            INFO: highest_uniques(): Yes, it's a bad tile\033[m" )
      
  return IsBadTile
  
# ------------------------------------------------------------------------------

def highest_uniques(args, oslide, level, slide_width, slide_height, tile_size, samples, n_tiles):

  x_high=0
  y_high=0
  uniques=0
  high_uniques=0
  second_high_uniques=0
  excellent_starting_point_found  = False
  reasonable_starting_point_found = False
   
  scan_range=int(n_tiles**0.5)
  
  if (DEBUG>99):
    print ( f"TILER:            INFO: highest_uniques(): scan_range = {scan_range}" )
  
  for n in range(0, samples):
  
    x = randint( 1, (slide_width  - tile_size)) 
    y = randint( 1, (slide_height - tile_size)) 
                    
    tile = oslide.read_region((x, y), level, ( tile_size, tile_size) );
    IsBadTile=check_badness( args, tile )
    if IsBadTile:
      pass
    
    uniques = len(np.unique(tile ))

    if (DEBUG>99):
      print ( f"TILER:            INFO: uniques(): (n={n:3d}) a tile with \r\033[68C{GREEN}{uniques:4d}{RESET} uniques at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}" )


    if ( uniques>high_uniques ):                                                                                    # then check the tiles at the other three corners of the putative sqaure

      badness_count=0
      
      IsBadTile=False
      tile_south = oslide.read_region((x,                          y+(scan_range-1)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_south )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_east = oslide.read_region((x+(scan_range-1)*tile_size,      y),                      level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_east )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_southeast = oslide.read_region((x+(scan_range-1)*tile_size, y+(scan_range-1)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_southeast )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_centre = oslide.read_region((x+(scan_range//2)*tile_size, y+(scan_range//2)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_centre )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_inner_sw = oslide.read_region((x+(scan_range//4)*tile_size, y+(scan_range//4)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_inner_sw )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_inner_ssw = oslide.read_region((x+(scan_range//4)*tile_size, y+(scan_range//4)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_inner_ssw )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_outer_sw = oslide.read_region((x+(3*scan_range//4)*tile_size, y+(3*scan_range//4)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_outer_sw )
      if IsBadTile:
        badness_count+=1
        
      IsBadTile=False
      tile_outer_ssw = oslide.read_region((3*x+(scan_range//4)*tile_size, y+(3*scan_range//4)*tile_size), level, ( tile_size, tile_size) );
      IsBadTile=check_badness( args, tile_outer_ssw )
      if IsBadTile:
        badness_count+=1

      if badness_count==0:
        excellent_starting_point_found=True
        high_uniques=uniques
        x_high=x
        y_high=y
        if (DEBUG>3):
          print ( f"\033[1mTILER:            INFO: highest_uniques():     (n={n:3d}) a tile with \r\033[62C{GREEN}{high_uniques:4d}{RESET} unique colour values (proxy for information content) and bad-corner-tile-count= \r\033[146C{CYAN}{badness_count}{RESET} was found at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}\033[m" )
        if high_uniques>=240:                                                                               # Max possible value is 256, so let's call 240 or better good enough
          break        
      elif badness_count<=4:
        if ( uniques>second_high_uniques ): 
          reasonable_starting_point_found=True
          second_high_uniques=uniques
          x2_high=x
          y2_high=y
          if excellent_starting_point_found==False:
            if (DEBUG>3):
              print ( f"TILER:            INFO: second_high_uniques(): (n={n:3d}) a tile with \r\033[62C{GREEN}{second_high_uniques:4d}{RESET} unique colour values (proxy for information content) and bad-corner-tile count= \r\033[146C{CYAN}{badness_count}{RESET} was found at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}" )
      
  if excellent_starting_point_found==True:
    return ( x_high, y_high, high_uniques )
  elif reasonable_starting_point_found==True:
    return ( x2_high, y2_high, second_high_uniques )
  else:
    return ( x,      y,      999 )        
      

# ------------------------------------------------------------------------------

def choose_mag_level( my_thread, zoom_out_prob, zoom_out_mags, r_norm ):

   
  if len(zoom_out_prob)!=len(zoom_out_mags)!=1:
    print( f"\r{RESET}{RED}TILER:     FATAL: configuration vectors '{CYAN}zoom_out_prob{RESET}{RED}' and '{CYAN}zoom_out_mags{RESET}{RED}' have differing numbers of entries ({MIKADO}{len(zoom_out_prob)}{RESET}{RED} and {MIKADO}{len(zoom_out_mags)}{RESET}{RED} entries respectively){RESET}", flush=True)
    print( f"\r{RESET}{RED}TILER:     FATAL: ... halting now{RESET}" )
    sys.exit(0)
  
  if sum(zoom_out_prob)==0:                                                                  # then user wants zoom_out_prob to be random
    
    if DEBUG>3:  
      print( f'\r{RESET}TILER:          INFO: user wants system to generate {CYAN}zoom_out_prob vector{RESET}', end='', flush=True  )

    
    multiplier = int(np.random.choice(
      zoom_out_mags, 
      1,
      p=r_norm
    ))
    
    if DEBUG>2:
      print( f'\r{RESET}TILER:          INFO: system generated {CYAN}zoom_out_prob vector{RESET} = {ASPARAGUS}{r_norm}{RESET}', end='', flush=True  )

  
  else:
  
    if DEBUG>2:  
      print( f'\r{RESET}TILER:          INFO: user supplied  {CYAN}zoom_out_prob vector{RESET} = {CHARTREUSE}{zoom_out_prob}{RESET}', end='', flush=True )
  
      
    r      = [ random.random() for i in range(1, len(zoom_out_prob)+1 ) ]
    r_norm = [ i/(sum(r)) for i in r ]                                                                       # make the vector add up to 1
    
    multiplier = int(np.random.choice(
      zoom_out_mags, 
      1,
      p=zoom_out_prob
    ))
    
    
  return multiplier

# ------------------------------------------------------------------------------

# ~ def show_image ( image ):

    # ~ width, height = image.size

    # ~ if DEBUG>4:
      # ~ print ( f"P_C_DATASET:        INFO:    show_image()                   type( image)   = {CAMEL}{   type( image)  }{RESET}"   )
      # ~ print ( f"P_C_DATASET:        INFO:    show_image()                   width/height       = {CAMEL}{    image.size   }{RESET}"   )
      # ~ print ( f"P_C_DATASET:        INFO:    show_image()                   channels           = {CAMEL}{    image.mode   }{RESET}"   )
    
    # ~ root = Tk()
    # ~ screen_resolution = str(width)+'x'+str(height)  
    # ~ root.geometry(screen_resolution)
    # ~ canvas = Canvas(root,width=width, height=height)
    # ~ canvas.pack()
    # ~ image = ImageTk.PhotoImage( image )
    # ~ imagesprite = canvas.create_image( height/2, width/2, image=image, )
    # ~ root.mainloop()
