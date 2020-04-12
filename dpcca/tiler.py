"""
This routine performs tiling for exactly one SVS file

"""

import os
import sys
import cv2
import time
import glob
import random
import psutil
import argparse
import multiprocessing
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**32)
import openslide
import numpy as np
import tkinter as tk
from tkinter            import Label, Tk
from random             import randint
from norms              import Normalizer, NormalizerNone, NormalizerReinhard, NormalizerSPCN
from PIL                import ImageTk
from PIL                import Image
from shutil             import copyfile as cp
from scipy.stats.mstats import ttest_1samp
from torchvision        import transforms

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=350)


BB="\033[35;1m"
RESET="\033[m"

CYAN='\033[36;1m'
RED='\033[31;1m'
GREEN='\033[32;1m'

DEBUG=1

num_cpus = multiprocessing.cpu_count()

def tiler( args, n_tiles, batch_size, stain_norm, norm_method, d, f, my_thread ):


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
  
  # DON'T USE args.n_tiles since it is the job level list of numbers of tiles. Here we are just using one value of n_tiles, passed in as the parameter above  
  just_profile           = args.just_profile                                                                # display an analysis of image tiles then exit
  just_test              = args.just_test                                                                   # if set, suppress tile quality filters (i.e. accept every tile)
  data_dir               = args.data_dir
  rand_tiles             = args.rand_tiles                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
  tile_size              = args.tile_size                                                                   # if not 0, size of tile to be generated (e.g. for dpccaI need to be able to set an absolute tile size)
  greyness               = args.greyness                                                                    # Used to filter out images with very low information value
  min_uniques            = args.min_uniques                                                                 # tile must have at least this many unique values or it will be assumed to be degenerate
  min_tile_sd            = args.min_tile_sd                                                                 # Used to cull slides with a very reduced greyscale palette such as background tiles 
  points_to_sample       = args.points_to_sample                                                            # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination

  if just_test=='True':
    greyness=0
    min_uniques=0
    min_tile_sd=2
    points_to_sample=100
    

  if not just_profile=='True':
    if (DEBUG>0):
      print ( f"process/slide: {BB}{my_thread:2d}) {f:66s}{RESET} ", flush=True, end="")
  already_displayed=False
      
  if (DEBUG>9):
    print ( f"TILER: INFO: (parent directory)  = {BB}{f}{RESET}",                  flush=True)
    print ( f"TILER: INFO: (thread num)        = {BB}{my_thread}{RESET}",          flush=True)
    print ( f"TILER: INFO: (stain_norm)        = {BB}{stain_norm}{RESET}",         flush=True)
    print ( f"TILER: INFO: (thread num)        = {BB}{stain_norm_target}{RESET}",  flush=True)

  fqn = f"{data_dir}/{d}/{f}"
  
  ALLOW_REDUCED_WIDTH_EDGE_TILES = 0                                                                       # if 1, then smaller tiles will be generated, if required, at the right hand edge and bottom of the image to ensure that all of the image is tiled
  
  level            = 0
  tile_size_40X    = 2100;                                                                                 # only used if resizing is enabled (tile_size=0)
  
  start = time.time()
  
  if (DEBUG>0):  
    print('TILER: INFO: now processing          {:}{:}{:}'.format( BB, fqn, RESET));
  
  try:
    oslide = openslide.OpenSlide( fqn );                                                                   # open the file containing the image
    
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in oslide.properties:                                       # microns per pixel that the image was scanned at
        if (DEBUG>9):
          print('TILER: INFO: OBJECTIVE POWER      = {:}{:}{:}'.format(BB, oslide.properties[ openslide.PROPERTY_NAME_OBJECTIVE_POWER], RESET )  ) 
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:                                                 # microns per pixel that the image was scanned at
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if (DEBUG>9):
          print('TILER: INFO: MICRONS/PXL (X)      = {:}{:}{:}'.format(BB, oslide.properties[openslide.PROPERTY_NAME_MPP_X], RESET )  )
          print('TILER: INFO: mag                  = {:}{:}/{:} = {:0.2f}{:}'.format(BB, 10.0, float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]), mag, RESET ))
    elif "XResolution" in oslide.properties:                                                               # for TIFF format images (apparently)  https://openslide.org/docs/properties/
        mag = 10.0 / float(oslide.properties["XResolution"]);
        if (DEBUG>9):
          print('TILER: INFO: XResolution      = {:}{:}{:} '.format(BB, oslide.properties["XResolution"], RESET )  )
          print('TILER: INFO: mag {:}{:}/{:}      = {:0.2f}{:} '.format(BB, 10.0, float(oslide.properties["XResolution"]), mag, RESET ) )
    else:
        mag = 10.0 / float(0.254);                                                                         # default, if we there is no resolution metadata in the slide, then assume it is 40x
        if (DEBUG>9):
          print('TILER: INFO: No openslide resolution metadata for this slide')
          print('TILER: INFO: setting mag to 10/.254      = {:}{:0.2f}{:}'.format( BB, (10.0/float(0.254)), RESET ))

    if (tile_size==0):                                                                                     # PGD 191217
      tile_width = int(tile_size_40X * mag / 40);                                                          # scale tile size from 40X to 'mag'. 'tile_size_40X' is set above to be 2100
    else:                                                                                                  # PGD 191217
      tile_width = tile_size                                                                               # PGD 191231
      
    width  = oslide.dimensions[0];                                                                       # width  of slide image
    height = oslide.dimensions[1];                                                                       # height of slide image

  except Exception as e:
    print('TILER:                   ERROR: exception!      {:}{:}{:}'.format(BB, e, RESET ) );
    exit(1);

  if (DEBUG>0):
    potential_tiles = (width-tile_width)*(height-tile_width) // (tile_width*tile_width)
    if not just_profile=='True':
      print( f"\r\033[KTILER:            INFO: slide height x width (pixels) = {BB}{height:6d} x {width:6d}{RESET} and potential {BB}{tile_width:3d} x {tile_width:3d}{RESET} tiles for this slide = {BB}{potential_tiles:7d}{RESET}", flush=True )

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
   
  if just_test=='True':
    if DEBUG>0:
      print( f"\033[1mTILER:            INFO: about to determine coordinates of patch in slide with high nominal contrast to use as starting coordinates for tiling \033[m" )  
    high_uniques=0
    x_start, y_start, high_uniques = highest_uniques( args, oslide, level, width, height, tile_width )
    if high_uniques==0:                                                                                    # means we went found no qualifying tile to define the patch by (can happen)
      x_start=int( width//2)
      y_start=int(height//2)
      print( f"\033[38;2;255;165;0m\033[1mTILER:            INFO:  no suitable patch found: setting coordinates to centre of slide x={x_start:7d} y={y_start:7d}\033[m" )
    else:
      if DEBUG>0:
        print( f"\033[1m\033[mTILER:            INFO:  coordinates of tile with best contrast: x={x_start:7d} y={y_start:7d} and highest number of unique RGB values = {high_uniques:5d}\033[m" )
  
  if just_test=="False":
    x_start=0
    y_start=0
  else:
      print( f"\033[31;1mTILER:            INFO:  CAUTION! 'just_test' flag is set. Tiling will commence at these coordinates, selected for good contrast: {x_start},{y_start} \033[m" )  
  
  to_get=int(batch_size**0.5)
  
  break_now=False
  
  if just_test=='False':
    x_span=range(x_start, width, tile_width)                                                               # steps of tile_width
    y_span=range(y_start, width, tile_width)                                                               # steps of tile_width
  else:
    x_span=range(x_start, x_start + (to_get*tile_width) , tile_width)                                         # steps of tile_width
    y_span=range(y_start, y_start + (to_get*tile_width) , tile_width)                                         # steps of tile_width
    
  if DEBUG>99:
    print( f"\033[1mTILER:            INFO:  to_get={to_get}\033[m" )    
    print( f"\033[1mTILER:            INFO:  x_span={x_span}\033[m" )
    print( f"\033[1mTILER:            INFO:  y_span={y_span}\033[m" )
  
  for x in x_span:

      if break_now==True:
        break
      if ( tiles_processed>n_tiles ):
        break

      for y in y_span:
  
          tiles_considered_count+=1
            
          if ( tiles_processed>=n_tiles ):                                                                  # i.e. stop when we have the requested number of tiles
            if DEBUG>99:
              print ( f"tiles_processed = {BB}{tiles_processed}{RESET} ", flush=True)
            break_now=True
            break

          else:
            if (x>width-2*tile_width) & (y>height-2*tile_width):
              if just_profile=='True':
                if already_displayed==False:
                  print( f"\n\033[31m\033[1m slide {f} had insufficient tiles ({tiles_processed}, but {n_tiles} are required) that met the tile quality criteria. \033[m", flush=True)
                  already_displayed=True
              else:
                if just_test==False:
                  print('\n\033[31m\033[1mTILER: FATAL: For slide {:} at {:},{:} there are insufficient tiles (have {:}) that meet the chosen criteria. Halting this thread now\033[m'.format( fqn, x, y, tiles_processed ), flush=True)
                  return FAIL
              
            if x + tile_width > width:
                pass
            else:
                tile_width_x = tile_width;
            if y + tile_width > height:
                pass
            else:
                tile_width_y = tile_width;
                        
  
            x_resize = int(np.ceil(tile_size_40X * tile_width_x/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
            y_resize = int(np.ceil(tile_size_40X * tile_width_y/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
 

            x_rand = randint( 1, (width  - tile_width_x)) 
            y_rand = randint( 1, (height - tile_width_y)) 

            
            if just_test=='True':
              rand_tiles='False'
              
            if ( rand_tiles=='False'):                                                              
              if just_test==False:                                                                         # error message disabled if 'just_test' mode is enabled
                if (DEBUG>999):
                  print ( "TILER: INFO:  random tile selection has been disabled. It probably should be enabled ( --rand_tiles='True'" )
              tile = oslide.read_region((x,      y),      level, (tile_width_x, tile_width_y));                      # extract the tile from the slide. Returns an PIL RGBA Image object
              fname = '{0:}/{1:}/{2:06}_{3:06}.png'.format( data_dir, d, y, x)  # use the tile's top-left coordinate to construct a unique filename
            else:
              if (DEBUG>999):
                print ( "TILER: INFO:  random tile selection is enabled. Use switch --rand_tiles='False' in the unlikely event that you want to disable it" )
              tile = oslide.read_region((x_rand, y_rand), level, (tile_width_x, tile_width_y));            # extract the tile from a randon position on the slide. Returns an PIL RGBA Image object
              fname = '{0:}/{1:}/{2:06}_{3:06}.png'.format( data_dir, d, x_rand, y_rand)  # use the tile's top-left coordinate to construct a unique filename

            if (DEBUG>999):
              print ( "TILER: INFO:               tile = \033[1m{:}\033[m".format(np.array(tile)) )              

            if (tile_size==0):                                                                             # tile_size=0 means resizing is desired by user
              tile = tile.resize((x_resize, y_resize), Image.ANTIALIAS)                                    # resize the tile; use anti-aliasing option


            # decide by means of a heuristic whether the image contains is background or else contains too much background
            IsBackground   = check_background( args, tile )
            if IsBackground:
              background_image_count+=1
            
            # decide by means of a heuristic whether the image is of low contrast
            IsLowContrast = check_contrast   ( args, tile )
            if IsLowContrast:
              low_contrast_tile_count       +=1

            # check the number of unique values in the image, which we will use as a proxy to discover degenerate (images)
            IsDegenerate  = check_degeneracy ( args, tile )
            if IsDegenerate:
              degenerate_image_count+=1

            if ( IsBackground | IsDegenerate | IsLowContrast ) & ( just_test=='False' ):                   # If 'just_test' = True, all tiles must be accepted
              if (DEBUG>999):
                print ( "TILER: INFO:               skipping this tile" ) 
              pass
      

            else:
              if not stain_norm =="NONE":                                                                  # then perform the selected stain normalization technique on the tile W

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
              
              #if (DEBUG>9):
              #    print ( "TILER: INFO: saving   \r\033[65C\033[32m{:}\033[m, standard deviation = \033[32m{:>3.1f}\033[m".format( fname, sample_sd  ) )
              #if (DEBUG>9):oslide, width, height, tile_width
              #    print ( "TILER: INFO: saving   \r\033[65C\033[32m{:}\033[m with greyscale range = \033[32;1;4m{:}\033[m)".format( fname, greyscale_range) )

              if (DEBUG>9):
                print ( "TILER: INFO:               x = \033[1m{:}\033[m".format(x),             flush=True)
                print ( "TILER: INFO:               y = \033[1m{:}\033[m".format(y),             flush=True)  
                print ( "TILER: INFO:      tile_width = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:     tile_height = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:          fname  = \033[1m{:}\033[m".format( fname ) )

              tile.save(fname);                                                                            # save to the filename we made for this tile earlier              
              tiles_processed += 1
              
#             print ( "\033[s\033[{:};{:}f\033[32;1m{:}{:2d};{:>4d} \033[m\033[u".format( randint(1,68), int(1500/num_cpus)+7*my_thread, BB, my_thread+1, tiles_processed ), end="", flush=True )
              if (DEBUG>99):
                print ( f"\033[s\033[{tiles_processed//50};{int(1500/num_cpus)+7*my_thread}f\033[32;1m{BB}{my_thread+1:2d};{tiles_processed:>4d} \033[m\033[u", end="", flush=True )
    
  if just_profile=='True':
    print ( f"\
 \033[34mslide=\033[1m{f:66s}\033[m\
 \033[34mheightxwidth=\033[1m{height:6d} x{height:6d}\033[m\
 \033[34mavailable {tile_width_x:3d} x{tile_width_x:3d} tiles=\033[1m{potential_tiles:6d} \033[m\
 \033[34manalysed=\033[1m{tiles_considered_count:6d} \033[m\
 \033[34macceptable=\033[1m{tiles_processed:5d} \
 \033[1m({tiles_processed/tiles_considered_count *100:2.0f})%\033[m\
 \033[34mlow contrast=\033[1m{low_contrast_tile_count:5d};\
 \033[1m({low_contrast_tile_count/tiles_considered_count *100:2.1f})%\033[m\
 \033[34mdegenerate=\033[1m{degenerate_image_count:5d} \
 \033[1m({degenerate_image_count/tiles_considered_count *100:2.1f}%)\033[m\
 \033[34mbackground=\033[1m{background_image_count:5d} \
 \033[1m({background_image_count/tiles_considered_count *100:2.0f})% \033[m", flush=True )
  else:
    if (DEBUG>0):
      print ( f"\033[s\033[{my_thread+15};1f\
 \033[34mt=\033[1m{my_thread:>2d}\033[m\
 \033[34mc=\033[1m{tiles_considered_count:4d} \033[m\
 \033[34mok=\033[1m{tiles_processed:4d} \
 \033[1m{tiles_processed/tiles_considered_count *100:2.0f}%\033[m\
 \033[34mgr=\033[1m{low_contrast_tile_count:3d};\
 \033[1m{low_contrast_tile_count/tiles_considered_count *100:2.1f}%\033[m\
 \033[34mdeg=\033[1m{degenerate_image_count:4d} \
 \033[1m{degenerate_image_count/tiles_considered_count *100:2.1f}%\033[m\
 \033[34mbkg=\033[1m{background_image_count:4d} \
 \033[1m{background_image_count/tiles_considered_count *100:2.0f}% \033[m\
\033[u", flush=True, end="" ) 
  
  if (DEBUG>9):
    print('TILER: INFO: time taken to tile this SVS image: \033[1m{0:.2f}s\033[m'.format((time.time() - start)/60.0))

  if (DEBUG>9):
    print ( "TILER: INFO: about to display the \033[33;1m{:,}\033[m tiles".format    ( tiles_processed   ) )
    result = display_processed_tiles( data_dir, DEBUG )

    
  return SUCCESS

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def button_click_exit_mainloop (event):
    event.widget.quit()                                                                                    # this will cause mainloop to unblock.

# ------------------------------------------------------------------------------

def display_processed_tiles( the_dir, DEBUG ):

# from: https://code.activestate.com/recipes/521918-pil-and-tkinter-to-display-images/

  if (DEBUG>9):
    print ( "TILER: INFO: at top of display_processed_tiles() and dir = \033[33;1m{:}\033[m".format( the_dir   ) )

  dirlist         = os.listdir( the_dir )

  for f in dirlist:
    if (DEBUG>9):
      print ( "TILER: INFO: display_processed_tiles() current file      = \033[33;1m{:}\033[m".format( f  ) )
      try:
          master = Tk()
          master.bind("<Button>", button_click_exit_mainloop )
          master.geometry('+%d+%d' % (1350,500))                                                           # set window position
          old_label_image = None
          image1 = Image.open(f)                                                                           # open the file
          resized = image1.resize((512, 512),Image.ANTIALIAS)
          master.geometry('%dx%d' % (resized.size[0],resized.size[1]))                                     # set the size to be the same dimensions as a tile
          tkpi = ImageTk.PhotoImage(resized)                                                               # convert the png image into a canonical tkinter image object (tkinter doesn't natively support png)
          label_image = tk.Label(master, image=tkpi)                                                       # create a tkinter 'Label' display object
          label_image.tkpi=tkpi
          label_image.pack()                                                                               # 
          #label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])                           #
          master.title(f)                                                                                  # use the file name as the image title
          if old_label_image is not None:
              old_label_image.destroy()
          old_label_image = label_image
          master.mainloop()                                                                                # wait for user input (which we simulate with button_click_exit_mainloop())
      except Exception as e:
          if (DEBUG>9):
            print ( "TILER: INFO: Exception                                   = {:}".format( e  ) )
          # skip anything not an image
          # Warning, this will hide other errors as well
          pass

  return SUCCESS
                  

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

  tile_grey     = tile.convert('L')                                                            # make a greyscale copy of the image
  np_tile_grey  = np.array(tile_grey)
  
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

    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a background tile" )
    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a background tile" )

  return IsBackground

# ------------------------------------------------------------------------------
def check_contrast( args, tile ):

  # check greyscale range, as a proxy for useful information content
  tile_grey     = tile.convert('L')                                                                        # make a greyscale copy of the image
  greyscale_range  = np.max(np.array(tile_grey)) - np.min(np.array(tile_grey))                             # calculate the range of the greyscale copy
  GreyscaleRangeOk  = greyscale_range>args.greyness
  GreyscaleRangeBad = not GreyscaleRangeOk
  
  if GreyscaleRangeBad:
    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a low contrast tile" )
      
  return GreyscaleRangeBad
  
# ------------------------------------------------------------------------------
def check_degeneracy( args, tile ):

  # check number of unique values in the image, which we will use as a proxy to discover degenerate (articial) images
  unique_values = len(np.unique(tile )) 
  if (DEBUG>9):
    print ( "TILER:            INFO: highest_uniques(): number of unique values in this tile = \033[94;1;4m{:>3}\033[m) and minimum required is \033[94;1;4m{:>3}\033[m)".format( unique_values, min_uniques ) )
  IsDegenerate = unique_values<args.min_uniques

  if IsDegenerate:
    if (DEBUG>999):
      print ( f"TILER:            INFO: highest_uniques(): Yes, it's a degenerate tile" )
      
  return IsDegenerate
  
# ------------------------------------------------------------------------------
def check_badness( args, tile ):

  # check number of unique values in the image, which we will use as a proxy to discover degenerate (articial) images

  IsDegenerate  = check_degeneracy (args, tile)
  IsLowContrast = check_contrast   (args, tile)
  IsBackground  = check_background (args, tile)
  
  IsBadTile = IsDegenerate | IsLowContrast | IsBackground
  
  if IsBadTile:
    if (DEBUG>999):
      print ( f"\033[1mTILER:            INFO: highest_uniques(): Yes, it's a bad tile\033[m" )
      
  return IsBadTile
  
# ------------------------------------------------------------------------------

def highest_uniques(args, oslide, level, slide_width, slide_height, tile_size):

  x_high=0
  y_high=0
  uniques=0
  high_uniques=0
  second_high_uniques=0
  excellent_starting_point_found  = False
  reasonable_starting_point_found = False
  
  scan_range=int(args.n_tiles[0]**0.5)
  
  if (DEBUG>0):
    print ( f"TILER:            INFO: highest_uniques(): scan_range = {scan_range}" )
  
  break_now=False
  
  for n in range(0, 400):
  
    x = randint( 1, (slide_width  - tile_size)) 
    y = randint( 1, (slide_height - tile_size)) 
                    
    tile = oslide.read_region((x, y), level, ( tile_size, tile_size) );
    IsBadTile=check_badness( args, tile )
    if IsBadTile:
      pass
    
    uniques = len(np.unique(tile ))

    if (DEBUG>99):
      print ( f"TILER:            INFO: uniques(): (n={n:3d}) a tile with \r\033[62C{GREEN}{uniques:4d}{RESET} at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}" )


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
        if (DEBUG>0):
          print ( f"\033[1mTILER:            INFO: highest_uniques():     (n={n:3d}) a tile with \r\033[62C{GREEN}{high_uniques:4d}{RESET} unique colour values (proxy for information content) and bad corner tile count= \r\033[146C{CYAN}{badness_count}{RESET} was found at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}\033[m" )
      elif badness_count<=1:
        if ( uniques>second_high_uniques ): 
          reasonable_starting_point_found=True
          second_high_uniques=uniques
          x2_high=x
          y2_high=y
          if excellent_starting_point_found==False:
            if (DEBUG>0):
              print ( f"TILER:            INFO: second_high_uniques(): (n={n:3d}) a tile with \r\033[62C{GREEN}{second_high_uniques:4d}{RESET} unique colour values (proxy for information content) and bad corner tile count= \r\033[146C{CYAN}{badness_count}{RESET} was found at x=\r\033[162C{CYAN}{x:7d}{RESET}, y=\r\033[172C{CYAN}{y:7d}{RESET}" )
      
  if excellent_starting_point_found==True:
    return ( x_high, y_high, high_uniques )
  elif reasonable_starting_point_found==True:
    return ( x2_high, y2_high, second_high_uniques )
  else:
    return ( x,      y,      999 )        
      
        
        
