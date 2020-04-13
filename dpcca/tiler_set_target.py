"""
This routine establishes the target for stain normalization based on the user provided image file

"""

import os
import sys
import cv2
import time
import glob
import random
import psutil
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**32)
import openslide
import numpy   as np
from random             import randint
from norms              import Normalizer, NormalizerNone, NormalizerReinhard, NormalizerSPCN
from PIL                import Image
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

from tiler import *

np.set_printoptions(edgeitems=500)
np.set_printoptions(linewidth=200)

BB="\033[35;1m"
RESET="\033[m"

DEBUG=1


def tiler_set_target( args, stain_norm, stain_norm_target, writer ):

  a = random.choice( range(150,255) )
  b = random.choice( range(50,225) )
  c = random.choice( range(50,225) )
  BB="\033[38;2;{:};{:};{:}m".format( a,b,c )
  
  data_dir             = args.data_dir
  target_tile_coords   = args.target_tile_coords
  tile_size            = args.tile_size  
      
  fqn = f"{data_dir}/{stain_norm_target}"

  if (DEBUG>0):
    print ( f"TILER_SET_TARGET: INFO: (data_dir)              = {BB}{data_dir}{RESET}",           flush=True)
    print ( f"TILER_SET_TARGET: INFO: (stain_norm_target)     = {BB}{stain_norm_target}{RESET}",  flush=True)
    print ( f"TILER_SET_TARGET: INFO: (fqn)                   = {BB}{fqn}{RESET}",                flush=True)


  try:
    oslide = openslide.OpenSlide( fqn );                                                                   # open the file containing the image
  except Exception as e:
    print( f"\033[31;1mTILER_SET_TARGET: FATAL: slide selected as stain normalization target, '\033[3m{fqn}\033[m\033[31;1m', does not exist ... halting now\033[m" )
    sys.exit(0)

  level  = oslide.level_count-1
  width  = oslide.dimensions[0];                                                                           # width  of slide image
  height = oslide.dimensions[1];                                                                           # height of slide image

  if (DEBUG>0):
    print ( f"TILER_SET_TARGET: INFO: image  (width x height) = {BB}{width} x {height}{RESET}",           flush=True)
    print ( f"TILER_SET_TARGET: INFO: target (width x height) = {BB}{tile_size} x {tile_size}{RESET}",    flush=True)
    print ( f"TILER_SET_TARGET: INFO: target tile coords      = {BB}{target_tile_coords}{RESET}",         flush=True)

    if DEBUG>0:
      print( f"\033[1mTILER:            INFO: about to determine coordinates of tile in slide with high nominal contrast to use as starting coordinates for tiling \033[m" )  
    high_uniques=0
    samples=10000
    x_start, y_start, high_uniques = highest_uniques( args, oslide, level, width, height, tile_size, samples )
    if high_uniques==0:                                                                                    # means we went found no qualifying tile to define the patch by (can happen)
      x_start=int( width//2)
      y_start=int(height//2)
      print( f"\033[38;2;255;165;0m\033[1mTILER:            INFO:  no suitable patch found: setting coordinates to centre of slide x={x_start:7d} y={y_start:7d}\033[m" )
    else:
      if DEBUG>0:
        print( f"\033[1m\033[mTILER:            INFO:  coordinates of tile with best contrast: x={x_start:7d} y={y_start:7d} and highest number of unique RGB values = {high_uniques:5d}\033[m" )

  tile = oslide.read_region( (x_start, y_start), level, (tile_size, tile_size) )    # extract tile from the slide. Returns an PIL RGBA Image object

  # Use this tile as the target for stain normalization
  #
  # xxx First way is to provide Normalizer with mean and std parameters
  # xxx Mean(r, g, b) = (0, 0, 0), Std(r, g, b) = (1, 1, 1) 
  # xxx normalization_target = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
  #
  # This --> Second way is to provide Normalizer with a target image. It will normalize all other tiles to match the target image
  
  tile_rgb     = tile.convert('RGB')
  tile_rgb_npy = (np.array(tile_rgb))
  normalization_target = tile_rgb_npy

  if (DEBUG>0):
    print ( f"TILER_SET_TARGET: INFO: about to call 'Normalizer' with stain_norm = '\033[35m{stain_norm}\033[m' and normalization_target extracted from '\033[35m{args.stain_norm_target}\033[m'", flush=True ) 

  norm_method = Normalizer( stain_norm, normalization_target )                             #  one of <reinhard, spcn>;  target: Path of target image to normalize images to OR normalization_parameters as per above

  if (DEBUG>0):
    print ( f"TILER_SET_TARGET: INFO: norm_method.method                         = \033[36m{norm_method.method}\033[m,  norm_method.normalizer = \033[36m{norm_method.normalizer}\033[m",   flush=True )

  # Display target tile in Tensorboard
  
  if (DEBUG>99):
    print ( f"TILER_SET_TARGET: INFO: target tile shape       = {BB}{np.array(tile).shape}{RESET}",                    flush=True)
    print ( f"TILER_SET_TARGET: INFO: type(tile)              = {BB}{type(tile)}{RESET}",                              flush=True)
  #if (DEBUG>0):
    #print ( f"TILER_SET_TARGET: INFO: tile                    = {BB}{tile}{RESET}",                                    flush=True)
    #print ( f"TILER_SET_TARGET: INFO: tile_RGB                = {BB}{tile_RGB}{RESET}",                                flush=True) 
    #print ( f"TILER_SET_TARGET: INFO: tile_np.shape           = {BB}{tile_np.shape}{RESET}",                           flush=True)  
    #print ( f"TILER_SET_TARGET: INFO: tile_np[:,170:237,2]    = \n{BB}{tile_np[:,300:600,2]}{RESET}",                   flush=True)          
  
  #image = mpimg.imread("/home/peter/git/pipeline/dataset/56fdf145-26c2-44d2-bf0e-67d8c8999a6e/000001_002817_128_128.png")
  #fig = plt.figure(facecolor='yellow')
  #ax = fig.add_subplot(111)
  #ax.set_xlim([0,1])
  #ax.set_yscale('log')
  #ax.set_xlabel('Normalized entropy')
  #ax.set_ylabel('Frequency (log)')

  fig = plt.figure(dpi=200, frameon=False, )
  plt.imshow( tile, extent=[0,1000,0,1000] )
  #plt.show
    
  writer.add_figure( 'Target Image for Stain Normalization', fig )


  return norm_method
