"""
This routine performs tiling for exactly one SVS file, provided as arguments argv[1]. Saves tiles to folder specified in argv[2]
"""

import os
import sys
import cv2
import time
import glob
import random
import datetime
import openslide
import numpy as np
from PIL    import Image
from shutil import copyfile as cp

BB="\033[35;1m"
RESET="\033[m"

    
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

 
DEBUG=99

#from stin_norm_python.color_normalize_single_folder import color_normalize_single_folder

slide_name       = sys.argv[1]                         # Just the filename (no path)
file_dir         = sys.argv[2]                         # just the save path (no filename)
my_thread_number = sys.argv[3]

if (DEBUG>0):
  print ( "\n    SAVE_SVS_TO_TILES.PY: INFO: argv[1] (slide_name) = {:}{:}{:}".format( BB, sys.argv[1], RESET ),  flush=True)
  print ( "    SAVE_SVS_TO_TILES.PY: INFO: argv[2] (file_dir)   = {:}{:}{:}".format( BB, sys.argv[2], RESET ),  flush=True)
  print ( "    SAVE_SVS_TO_TILES.PY: INFO: argv[3] (thread num) = {:}{:}{:}".format( BB, sys.argv[3], RESET ),  flush=True)	
  
INCLUDE_WHITE_TILES         = int  (sys.argv[7])                                                           # if 1, dummy white tiles will be generated; if 0, would-be white tiles will be ignored
WHITENING_THRESHOLD         = float(sys.argv[6])                                                           # threshold to determine whether a tile is 'white' 
TILE_SIZE                   = int  (sys.argv[5])                                                           # if not 0, size of tile to be generated (e.g. for dpccaI need to be able to set an absolute tile size)
TILES_TO_GENERATE_PER_SVS   = int  (sys.argv[4])                                                           # how many files to be GENERATED per image. Caution, because the selection will NOT be random. (NB: can decide how many to USE per image in generate.py with parm "MAX_ALLOWED_TILES_PER_SVS"

ALLOW_REDUCED_WIDTH_EDGE_TILES = 0                                                                         # if 1, then smaller tiles will be generated, if required, at the right hand edge and bottom of the image to ensure that all of the image is tiled

level            = 0
tile_size_40X    = 2100;                                                                                   # only used if resizing is enabled (TILE_SIZE=0)

start = time.time()
tiles_processed = 0

#fdone = '{}/extraction_done.txt'.format(output_folder);                                                   # leave a file in the folder to indicate that this SVS has been filed << PGD 191217 - should be at the end                                   

#if os.path.isfile(fdone):                                                                                 # skip if we've already done this slide
#    print('SAVE_SVS_TO_TILES: fdone {} exist, skipping'.format(fdone));
#    exit(0);

print('    SAVE_SVS_TO_TILES.PY: INFO: now extracting         {:}{:}{:}'.format( BB, slide_name, RESET));

try:
    oslide = openslide.OpenSlide(slide_name);                                                              # open the file containing the image
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in oslide.properties:                                       # microns per pixel that the image was scanned at
        if (DEBUG>0):
          print('    SAVE_SVS_TO_TILES.PY: INFO: OBJECTIVE POWER      = {:}{:}{:}'.format(BB, oslide.properties[ openslide.PROPERTY_NAME_OBJECTIVE_POWER], RESET )  ) 
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:                                                 # microns per pixel that the image was scanned at
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if (DEBUG>0):
          print('    SAVE_SVS_TO_TILES.PY: INFO: MICRONS/PXL (X)      = {:}{:}{:}'.format(BB, oslide.properties[openslide.PROPERTY_NAME_MPP_X], RESET )  )
          print('    SAVE_SVS_TO_TILES.PY: INFO: mag                  = {:}{:}/{:} = {:0.2f}{:}'.format(BB, 10.0, float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]), mag, RESET ))
    elif "XResolution" in oslide.properties:                                                               # for TIFF format images (apparently)  https://openslide.org/docs/properties/
        mag = 10.0 / float(oslide.properties["XResolution"]);
        if (DEBUG>0):
          print('    SAVE_SVS_TO_TILES.PY: INFO: XResolution      = {:}{:}{:} '.format(BB, oslide.properties["XResolution"], RESET )  )
          print('    SAVE_SVS_TO_TILES.PY: INFO: mag {:}{:}/{:}      = {:0.2f}{:} '.format(BB, 10.0, float(oslide.properties["XResolution"]), mag, RESET ) )
    else:
        mag = 10.0 / float(0.254);                                                                         # default, if we there is no resolution metadata in the slide, then assume it is 40x
        if (DEBUG>0):
          print('    SAVE_SVS_TO_TILES.PY: INFO: No openslide resolution metadata for this slide')
          print('    SAVE_SVS_TO_TILES.PY: INFO: setting mag to 10/.254      = {:}{:0.2f}{:}'.format( BB, (10.0/float(0.254)), RESET ))

    if (TILE_SIZE==0):                                                                                     # PGD 191217
      tile_width = int(tile_size_40X * mag / 40);                                                          # scale tile size from 40X to 'mag'. 'tile_size_40X' is set above to be 2100
    else:                                                                                                  # PGD 191217
      tile_width = TILE_SIZE                                                                               # PGD 191231
      
    width  = oslide.dimensions[0];                                                                         # width  of slide image
    height = oslide.dimensions[1];                                                                         # height of slide image
except:
    print('    SAVE_SVS_TO_TILES.PY: ERROR: {:}{:}{:}: exception caught:'.format(BB, slide_name, RESET ) );
    exit(1);

mask_file      = '{}{}_mask.png'.format(file_dir, slide_name[:-4])                                          #  reconstruct the name of the mask file

if (DEBUG>0):
  print('    SAVE_SVS_TO_TILES.PY: INFO: mask_file            = {:}{:}{:}'.format( BB, mask_file, RESET) )

mask           = cv2.imread(mask_file, 0)                                                                   # open the mask file
mask[mask > 0] = 1                                                                                          # turn it into a binary mask
scale          = height/mask.shape[0]                                                                       # calculate the relative scale of the slide compared to the mask file

print('    SAVE_SVS_TO_TILES.PY: INFO: slide height/width   = {:}{:}/{:}{:}'.format(BB, height, width, RESET))
print('    SAVE_SVS_TO_TILES.PY: INFO: mask size            = {:}{:}{:}'.format ( BB, mask.shape, RESET ) )


for x in range(1, width, tile_width):                                                                      # in steps of tile_width
	                                                                          
    for y in range(1, height, tile_width):                                                                 # in steps of tile_width

        if ( tiles_processed < TILES_TO_GENERATE_PER_SVS ):
		  
          if x + tile_width > width:
              if (ALLOW_REDUCED_WIDTH_EDGE_TILES==1):   
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
          
          isWhite = np.sum(mask[y_m:y_m + int(tile_width_y/scale), x_m:x_m + int(tile_width_x/scale)])/(tile_width_x*tile_width_y/scale/scale) > WHITENING_THRESHOLD    # is it a "white" tile (predominantly white) ?
          
          x_resize = int(np.ceil(tile_size_40X * tile_width_x/tile_width))                                 # only used if TILE_SIZE=0
          y_resize = int(np.ceil(tile_size_40X * tile_width_y/tile_width))                                 # only used if TILE_SIZE=0
  
          #print('x_resize/y_resize: {}/{}'.format(x_resize, y_resize))
          if isWhite:                                                                                      # if this tile meets the definition of a white tile (see WHITENING_THRESHOLD)
              if (INCLUDE_WHITE_TILES==1):
                # this is a white tile, generate dummy tile which is ALL white to avoid pointless computations later on
                if (TILE_SIZE==0):                                                                         # TILE_SIZE would be 0 if resizing is desired; otherwise TILE_SIZE is the size of tiles to be extracted
                  dummy = np.ones((y_resize, x_resize, 3))*255                                             # matrix with every value 255 (= white)
                else:
                  dummy = np.ones((TILE_SIZE, TILE_SIZE, 3))*255                                           # matrix with every value 255 (= white)
                dummy = dummy.astype(np.uint8)                                                             # convert to python short integers (PGD 191217 - could have done this in the last line)
                cv2.imwrite(fname, dummy)                                                                  # save to the filename we made for this tile
                tiles_processed += 1
          else:                                                                                            # it's a normal tile (i.e. has non-trivial information content)
              tile = oslide.read_region((x, y), level, (tile_width_x, tile_width_y));                      # extract the tile from the slide
              '''
              location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
              level (int) – the level number
              size (tuple) – (width, height) tuple giving the region size
              '''
              if (TILE_SIZE==0):                                                                           # TILE_SIZE would be 0 if resizing is desired; otherwise TILE_SIZE is the size of tiles to be extracted
                tile = tile.resize((x_resize, y_resize), Image.ANTIALIAS)                                  # resize the tile; use anti-aliasing option
                                                                                                           # save to the filename we made for this tile
              if (DEBUG>99):
                print ( "    SAVE_SVS_TO_TILES.PY: INFO:               x = \033[1m{:}\033[m".format(x),             flush=True)
                print ( "    SAVE_SVS_TO_TILES.PY: INFO:               y = \033[1m{:}\033[m".format(y),             flush=True)  
                print ( "    SAVE_SVS_TO_TILES.PY: INFO:      tile_width = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "    SAVE_SVS_TO_TILES.PY: INFO:     tile_height = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "    SAVE_SVS_TO_TILES.PY: INFO:          fname  = \033[1m{:}\033[m".format( fname ) )

              tile.save(fname);                 
              tiles_processed += 1
          
          if ( isWhite == False ):
            if ( tiles_processed%25==0 ):
              print('    SAVE_SVS_TO_TILES.PY: INFO: (thread|done|max)    =  {0:>02s}|{1:>4d}|{2:<4d} \r\033[70Cimage name: {3:s};  \r\033[149Ccurrent tile coordinates: {4:6d},{5:6d} \r\033[190Ctile height/width: {6:>4d}/{7:<4d}'.format(my_thread_number, tiles_processed, TILES_TO_GENERATE_PER_SVS, slide_name, x, y, tile_width, tile_width), flush=True)
          else:
            if (INCLUDE_WHITE_TILES==1):
              if ( tiles_processed%25==0 ):
                print('    SAVE_SVS_TO_TILES.PY: INFO: thread/processed/max = {:}/{:}/{:};  \r\033[63Ccurrent image name: {:};  \r\033[149Ccurrent tile coordinates: {:},{:}; \r\033[188Ctile height/width: {:}/{:} (White tile)'.format(my_thread_number, tiles_processed, TILES_TO_GENERATE_PER_SVS, x, y, slide_name, tile_width, tile_width), flush=True)

print('    SAVE_SVS_TO_TILES.PY: INFO: time taken to tile this SVS image: \033[1m{0:.2f}s\033[m'.format((time.time() - start)/60.0))
#open(fdone, 'w').close();

