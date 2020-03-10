"""
This routine performs tiling for exactly one SVS file, provided as arguments argv[1]. Saves tiles to folder specified in argv[2]

"""

import os
import sys
import cv2
import time
import glob
import random
import psutil
import argparse
import openslide
import numpy as np
import tkinter as tk
from tkinter import Label, Tk
from norms import Normalizer, NormalizerReinhard, NormalizerNone, NormalizerSPCN
from PIL     import ImageTk
from PIL     import Image
from shutil  import copyfile as cp

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=350)

BB="\033[35;1m"
RESET="\033[m"

DEBUG=1

a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )


def main(args):
 
  tiles_processed = 0
  tiles_available_count=0
  low_greyscale_range_tile_count=0
  
  #from stin_norm_python.color_normalize_single_folder import color_normalize_single_folder
  
  greyness = 25
  
  slide_name            = args.slide_name                                                                  # Just the slide's filename (no path). 'somefile.svs'
  file_dir              = args.file_dir                                                                    # just the save path (no filename)
  my_thread             = args.my_thread                                                                   # multiple instances of this program will run, each tackling some of the files according to a mod based on my_thread
  n_tiles               = args.n_tiles                                                                     # how many files to be GENERATED per image. Caution, because the selection will NOT be random. (NB: can decide how many to USE per image in generate.py with parm "MAX_ALLOWED_TILES_PER_SVS"
  tile_size             = args.tile_size                                                                   # if not 0, size of tile to be generated (e.g. for dpccaI need to be able to set an absolute tile size)
  whiteness             = args.whiteness                                                                   # threshold to determine whether a tile is 'white' 
  include_white_tiles   = args.include_white_tiles                                                         # if 1, dummy white tiles will be generated; if 0, would-be white tiles will be ignored
  greyness              = args.greyness                                                                    # Used to filter out images with very low information value
  colour_norm           = args.colour_norm                                                                 # if True, perform stain normalization (currently hard-wired to be "Reinhard" 

  if (DEBUG>0):
    print ( "\n    SAVE_SVS_TO_TILES.PY: INFO: (slide_name)          = {:}{:}{:}".format( BB, slide_name, RESET ),  flush=True)
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: (file_dir)            = {:}{:}{:}".format  ( BB, file_dir,   RESET ),  flush=True)
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: (thread num)          = {:}{:}{:}".format  ( BB, my_thread,  RESET ),  flush=True)	
  
  
  ALLOW_REDUCED_WIDTH_EDGE_TILES = 0                                                                       # if 1, then smaller tiles will be generated, if required, at the right hand edge and bottom of the image to ensure that all of the image is tiled
  
  level            = 0
  tile_size_40X    = 2100;                                                                                 # only used if resizing is enabled (tile_size=0)
  
  start = time.time()
  
  #fdone = '{}/extraction_done.txt'.format(output_folder);                                                 # leave a file in the folder to indicate that this SVS has been filed << PGD 191217 - should be at the end                                   
  
  #if os.path.isfile(fdone):                                                                               # skip if we've already done this slide
  #    print('SAVE_SVS_TO_TILES: fdone {} exist, skipping'.format(fdone));
  #    exit(0);
  
  print('    SAVE_SVS_TO_TILES.PY: INFO: now processing          {:}{:}{:}'.format( BB, slide_name, RESET));
  
  try:
      oslide = openslide.OpenSlide(slide_name);                                                            # open the file containing the image
      
      if openslide.PROPERTY_NAME_OBJECTIVE_POWER in oslide.properties:                                     # microns per pixel that the image was scanned at
          if (DEBUG>0):
            print('    SAVE_SVS_TO_TILES.PY: INFO: OBJECTIVE POWER      = {:}{:}{:}'.format(BB, oslide.properties[ openslide.PROPERTY_NAME_OBJECTIVE_POWER], RESET )  ) 
      if openslide.PROPERTY_NAME_MPP_X in oslide.properties:                                               # microns per pixel that the image was scanned at
          mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
          if (DEBUG>0):
            print('    SAVE_SVS_TO_TILES.PY: INFO: MICRONS/PXL (X)      = {:}{:}{:}'.format(BB, oslide.properties[openslide.PROPERTY_NAME_MPP_X], RESET )  )
            print('    SAVE_SVS_TO_TILES.PY: INFO: mag                  = {:}{:}/{:} = {:0.2f}{:}'.format(BB, 10.0, float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]), mag, RESET ))
      elif "XResolution" in oslide.properties:                                                             # for TIFF format images (apparently)  https://openslide.org/docs/properties/
          mag = 10.0 / float(oslide.properties["XResolution"]);
          if (DEBUG>0):
            print('    SAVE_SVS_TO_TILES.PY: INFO: XResolution      = {:}{:}{:} '.format(BB, oslide.properties["XResolution"], RESET )  )
            print('    SAVE_SVS_TO_TILES.PY: INFO: mag {:}{:}/{:}      = {:0.2f}{:} '.format(BB, 10.0, float(oslide.properties["XResolution"]), mag, RESET ) )
      else:
          mag = 10.0 / float(0.254);                                                                       # default, if we there is no resolution metadata in the slide, then assume it is 40x
          if (DEBUG>0):
            print('    SAVE_SVS_TO_TILES.PY: INFO: No openslide resolution metadata for this slide')
            print('    SAVE_SVS_TO_TILES.PY: INFO: setting mag to 10/.254      = {:}{:0.2f}{:}'.format( BB, (10.0/float(0.254)), RESET ))
  
      if (tile_size==0):                                                                                   # PGD 191217
        tile_width = int(tile_size_40X * mag / 40);                                                        # scale tile size from 40X to 'mag'. 'tile_size_40X' is set above to be 2100
      else:                                                                                                # PGD 191217
        tile_width = tile_size                                                                             # PGD 191231
        
      width  = oslide.dimensions[0];                                                                       # width  of slide image
      height = oslide.dimensions[1];                                                                       # height of slide image
  except Exception as e:
      print('    SAVE_SVS_TO_TILES.PY: ERROR: exception caught:      {:}{:}{:}'.format(BB, e, RESET ) );
      exit(1);
  
  mask_file      = '{}{}_mask.png'.format(file_dir, slide_name[:-4])                                       # reconstruct the name of the mask file
  
  if (DEBUG>0):
    print('    SAVE_SVS_TO_TILES.PY: INFO: mask_file            = {:}{:}{:}'.format( BB, mask_file, RESET) )
  
  mask           = cv2.imread(mask_file, 0)                                                                 # open the mask file
  mask[mask > 0] = 1                                                                                        # turn it into a binary mask
  scale          = height/mask.shape[0]                                                                     # calculate the relative scale of the slide compared to the mask file
  
  print('    SAVE_SVS_TO_TILES.PY: INFO: slide height/width   = {:}{:}/{:}{:}'.format(BB, height, width, RESET))
  print('    SAVE_SVS_TO_TILES.PY: INFO: mask size            = {:}{:}{:}'.format ( BB, mask.shape, RESET ) )
  
  
  for x in range(1, width, tile_width):                                                                    # in steps of tile_width
                                                                            
      for y in range(1, height, tile_width):                                                               # in steps of tile_width
  
          tiles_available_count+=1
  
          if ( tiles_processed < n_tiles ):                                               # i.e. stop when we have the requested number of tiles
  
            if x + tile_width > width:
                if (ALLOW_REDUCED_WIDTH_EDGE_TILES==1):                                                     # don't permit this. we don't need it.
                  if (DEBUG>9):
                    print('\033[31m\033[1m    SAVE_SVS_TO_TILES.PY: INFO: tile would go over right hand edge of image. Will reduce tile width to {:} to prevent this\033[37m\033[m'.format(width-x))
                  tile_width_x = width - x;                                                                  # left-fit the right-most tile should it go over the right edge
            else:
                tile_width_x = tile_width;
            if y + tile_width > height:
                if (ALLOW_REDUCED_WIDTH_EDGE_TILES==1):   
                  if (DEBUG>9):
                    print('\033[31m\033[1m    SAVE_SVS_TO_TILES.PY: INFO: tile would go over bottom edge of image. Will reduce tile height to {:} to prevent this\033[37m\033[m'.format(height-y))
                  tile_width_y = height - y;                                                                 # bottom-fit the bottom-most tile should it go over the bottom edge
            else:
                tile_width_y = tile_width;
                        
            fname = '{}/{}_{}_{}_{}.png'.format( file_dir, x, y, tile_width, tile_width);                # use the tile's top-left coordinate to construct a unique filename
  
    
            x_m = int(x/scale)
            y_m = int(y/scale)
            
            isWhite       = np.sum( mask[ y_m:y_m + int(tile_width_y/scale), x_m:x_m + int(tile_width_x/scale) ]) /  (tile_width_x*tile_width_y/scale/scale)  > whiteness    # is it a "white" tile (predominantly white)?
            
            x_resize = int(np.ceil(tile_size_40X * tile_width_x/tile_width))                                 # only used if tile_size=0, user flag to indicate that resizing is required
            y_resize = int(np.ceil(tile_size_40X * tile_width_y/tile_width))                                 # only used if tile_size=0, user flag to indicate that resizing is required
    
            #print('x_resize/y_resize: {}/{}'.format(x_resize, y_resize))
            if isWhite:                                                                                      # if this tile meets the definition of a white tile (see whiteness)
                if (include_white_tiles==1):
                  # this is a white tile, generate dummy tile which is ALL white to avoid pointless computations later on
                  if (tile_size==0):                                                                         # tile_size would be 0 if resizing is desired; otherwise tile_size is the size of tiles to be extracted
                    dummy = np.ones((y_resize, x_resize, 3))*255                                             # matrix with every value 255 (= white)
                  else:
                    dummy = np.ones((tile_size, tile_size, 3))*255                                           # matrix with every value 255 (= white)
                  dummy = dummy.astype(np.uint8)                                                             # convert to python short integers (PGD 191217 - could have done this in the last line)
                  cv2.imwrite(fname, dummy)                                                                  # save to the filename we made for this tile
                  tiles_processed += 1

            else:                                                                                            # it's a normal tile (i.e. has non-trivial information content)
  
                tile = oslide.read_region((x, y), level, (tile_width_x, tile_width_y));                      # extract the tile from the slide. Returns an PIL RGBA Image object
                '''
                location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
                level    (int)   – the level number
                size     (tuple) – (width, height) tuple giving the region size
                '''
                if (DEBUG>999):
                  print ( "    SAVE_SVS_TO_TILES.PY: INFO:               tile = \033[1m{:}\033[m".format(np.array(tile)) )              
  

                if (tile_size==0):                                                                           # tile_size would only be 0 if resizing is desired by user; (user flag to indicate that resizing is required)
                  tile = tile.resize((x_resize, y_resize), Image.ANTIALIAS)                                  # resize the tile; use anti-aliasing option
  
  
                # check greyscale range, as a proxy for useful information content
                tile_grey     = tile.convert('L')                                                            # make a greyscale copy of the image
                greyscale_range  = np.max(np.array(tile_grey)) - np.min(np.array(tile_grey))                 # calculate the range of the greyscale copy
  
                GreyscaleRangeOk  = greyscale_range>greyness
                GreyscaleRangeBad = not GreyscaleRangeOk
                
                if (DEBUG>1):
                  if GreyscaleRangeBad:
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:  skipping   \033[31m{:}\033[m with greyscale range = \033[31;1;4m{:}\033[m (minimum permitted is \033[31;1;4m{:}\033[m)".format( fname, greyscale_range, greyness)  )
                  if (DEBUG>999):
                    print ( "\n\n    SAVE_SVS_TO_TILES.PY: INFO:               grey_scaled tile shape                   = \033[1m{:}\033[m".format(np.array(tile_grey).shape) )
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:               grey_scaled tile                         = \n\033[1m{:}\033[m".format(np.array(tile_grey)) )              
  
                if GreyscaleRangeBad:                                                                      # skip low information tiles
                  low_greyscale_range_tile_count+=1

                else:                                                                                      # tile greyscale range is accepptable (presumed to have non-trivial information content)                                                                         

                  if (DEBUG>9):
                      print ( "    SAVE_SVS_TO_TILES.PY: INFO:  saving   \033[32m{:}\033[m with greyscale range = \033[32;1;4m{:}\033[m)".format( fname, greyscale_range) )

                  if (DEBUG>99):
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:               x = \033[1m{:}\033[m".format(x),             flush=True)
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:               y = \033[1m{:}\033[m".format(y),             flush=True)  
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:      tile_width = \033[1m{:}\033[m".format(tile_width),    flush=True)
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:     tile_height = \033[1m{:}\033[m".format(tile_width),    flush=True)
                    print ( "    SAVE_SVS_TO_TILES.PY: INFO:          fname  = \033[1m{:}\033[m".format( fname ) )
    
                  tile.save(fname);                                                                           # save to the filename we made for this tile earlier              
                  tiles_processed += 1
            
            
            
                  if not colour_norm =="NONE":

                    if (DEBUG>1):
                      print ( "    SAVE_SVS_TO_TILES.PY: INFO:  performing \033[36m{:}\033[m colour normalization on tile \033[36m{:}\033[m".format    ( colour_norm, fname  ) )

                      # Example of giving a parameter. Mean(r, g, b) = (0, 0, 0), Std(r, g, b) = (1, 1, 1)
                      parameter = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
                      normalizer = Normalizer(colour_norm, fname)
              
      
  
  if (DEBUG>999):
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: tiles available in image                       = \033[1m{:,}\033[m".format    ( tiles_available_count                       ) )
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: tiles   used                                   = \033[1m{:}\033[m".format     ( tiles_processed                             ) )
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: percent used                                   = \033[1m{:.2f}%\033[m".format ( tiles_processed/tiles_available_count *100  ) )
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: \033[31;1mlow greyscale range tiles (not used)           = {:}\033[m".format  ( low_greyscale_range_tile_count              ) )
  
  if (DEBUG>0):
    print('    SAVE_SVS_TO_TILES.PY: INFO: time taken to tile this SVS image: \033[1m{0:.2f}s\033[m'.format((time.time() - start)/60.0))


  if (DEBUG>9):
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: about to display the \033[33;1m{:,}\033[m tiles".format    ( tiles_processed   ) )
    SUCCESS = display_processed_tiles( file_dir, DEBUG )




# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def button_click_exit_mainloop (event):
    event.widget.quit()                                                                                   # this will cause mainloop to unblock.

# ------------------------------------------------------------------------------

def display_processed_tiles( the_dir, DEBUG ):

# from: https://code.activestate.com/recipes/521918-pil-and-tkinter-to-display-images/

  if (DEBUG>9):
    print ( "    SAVE_SVS_TO_TILES.PY: INFO: at top of display_processed_tiles() and dir = \033[33;1m{:}\033[m".format( the_dir   ) )

  dirlist         = os.listdir( the_dir )

  for f in dirlist:
    if (DEBUG>9):
      print ( "    SAVE_SVS_TO_TILES.PY: INFO: display_processed_tiles() current file      = \033[33;1m{:}\033[m".format( f  ) )
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
            print ( "    SAVE_SVS_TO_TILES.PY: INFO: Exception                                   = {:}".format( e  ) )
          # skip anything not an image
          # Warning, this will hide other errors as well
          pass

  return(1)
                  

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--slide_name',          type=str,   default='MISSING_SLIDE_NAME')
    p.add_argument('--file_dir',            type=str,   default='MISSING_DIRECTORY_NAME')
    p.add_argument('--my_thread',           type=int,   default=0)
    p.add_argument('--n_tiles',             type=int,   default=100)
    p.add_argument('--tile_size',           type=int,   default=128)
    p.add_argument('--whiteness',           type=float, default=0.1)
    p.add_argument('--include_white_tiles', type=int,   default=0)
    p.add_argument('--greyness',            type=int,   default=39)
    p.add_argument('--colour_norm',         type=str,   default='reinhard')

    args, _ = p.parse_known_args()

    main( args )

