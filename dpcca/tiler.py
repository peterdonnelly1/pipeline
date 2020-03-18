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
import numpy   as np
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

DEBUG=1

num_cpus = multiprocessing.cpu_count()

def tiler( args, stain_norm, d, f, my_thread ):

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
  
  greyness = 25
  
  data_dir              = args.data_dir
  n_tiles               = args.n_tiles                                                                     # how many tiles to be GENERATED per image
  rand_tiles            = args.rand_tiles                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
  tile_size             = args.tile_size                                                                   # if not 0, size of tile to be generated (e.g. for dpccaI need to be able to set an absolute tile size)
  greyness              = args.greyness                                                                    # Used to filter out images with very low information value
  min_uniques           = args.min_uniques                                                                 # tile must have at least this many unique values or it will be assumed to be degenerate
#  stain_norm            = args.stain_norm                                                                 # if True, perform stain normalization (currently hard-wired to be "Reinhard" 
  min_tile_sd           = args.min_tile_sd                                                                 # Used to cull slides with a very reduced greyscale palette such as background tiles 
  points_to_sample      = args.points_to_sample                                                            # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination

  if (DEBUG>0):
    print ( f"thread/slide: {BB}{my_thread}) {f}{RESET} ", flush=True, end="")
  if (DEBUG>1):
    print ( "TILER: INFO: (parent directory)  = {:}{:}{:}".format  ( BB, d, RESET ),  flush=True)
    print ( "TILER: INFO: (thread num)        = {:}{:}{:}".format  ( BB, my_thread,  RESET ),  flush=True)	

  fqn = f"{data_dir}/{d}/{f}"
  
  ALLOW_REDUCED_WIDTH_EDGE_TILES = 0                                                                       # if 1, then smaller tiles will be generated, if required, at the right hand edge and bottom of the image to ensure that all of the image is tiled
  
  level            = 0
  tile_size_40X    = 2100;                                                                                 # only used if resizing is enabled (tile_size=0)
  
  start = time.time()
  
  if (DEBUG>9):  
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
        if (DEBUG>0):
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

  if (DEBUG>9):  
    print('TILER: INFO: slide height/width   = {:}{:}/{:}{:}'.format(BB, height, width, RESET))

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

  
  for x in range(1, width, tile_width):                                                                    # in steps of tile_width

      if ( tiles_processed>=n_tiles ):
        break
                                                                                        
      for y in range(1, height, tile_width):                                                               # in steps of tile_width
  
          tiles_considered_count+=1
          
          if ( tiles_processed>=n_tiles ):
            break
            
          if ( tiles_processed<n_tiles ):                                                                  # i.e. stop when we have the requested number of tiles


            if (x>width-2*tile_width) & (y>height-2*tile_width):
              print('\033[31m\033[1mTILER: FATAL: For slide {:} at {:},{:} there are insufficient tiles (have {:}) that meet the chosen criteria. Halting this thread now\033[m'.format( pqn, x, y, tiles_processed ), flush=True)
              sys.exit(0)
              
            if x + tile_width > width:
                pass
            else:
                tile_width_x = tile_width;
            if y + tile_width > height:
                pass
            else:
                tile_width_y = tile_width;
                        
            fname = '{0:}/{1:}/{2:06}_{3:06}_{4:03}_{5:03}.png'.format( data_dir, d, x, y, tile_width, tile_width);  # use the tile's top-left coordinate to construct a unique filename
  
            x_resize = int(np.ceil(tile_size_40X * tile_width_x/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
            y_resize = int(np.ceil(tile_size_40X * tile_width_y/tile_width))                               # only used if tile_size=0, user flag to indicate that resizing is required
 

            x_rand = randint( 1, (width  - tile_width_x)) 
            y_rand = randint( 1, (height - tile_width_y)) 

            if rand_tiles=='False':
              if (DEBUG>999):
                print ( "TILER: INFO:  random tile selection has been disabled. It probably should be enabled ( --rand_tiles='True'" )
              tile = oslide.read_region((x,      y),      level, (tile_width_x, tile_width_y));                      # extract the tile from the slide. Returns an PIL RGBA Image object
            else:
              if (DEBUG>999):
                print ( "TILER: INFO:  random tile selection is enabled. Use switch --rand_tiles='False' in the unlikely event that you want to disable it" )
              tile = oslide.read_region((x_rand, y_rand), level, (tile_width_x, tile_width_y));            # extract the tile from a randon position on the slide. Returns an PIL RGBA Image object

            if (DEBUG>999):
              print ( "TILER: INFO:               tile = \033[1m{:}\033[m".format(np.array(tile)) )              

            if (tile_size==0):                                                                             # tile_size=0 means resizing is desired by user
              tile = tile.resize((x_resize, y_resize), Image.ANTIALIAS)                                    # resize the tile; use anti-aliasing option


            # decide by means of a heuristic whether the image contains too much background
            IsBackground   = check_background( tile,  points_to_sample, min_tile_sd )
            
            # decide by means of a heuristic whether the image contains too much background
            IsLowContrast = check_contrast   ( tile,  greyness )

            # check the number of unique values in the image, which we will use as a proxy to discover degenerate(images)
            IsDegenerate  = check_degeneracy ( tile,  min_uniques )

            if IsBackground:
              background_image_count+=1
              if (DEBUG>2):
                  print ( "TILER: INFO:  \r\033[32Cskipping mostly background tile \r\033[65C\033[94m{:}\033[m \r\033[162Cwith standard deviation =\033[94;1m{:>6.2f}\033[m (minimum permitted is \033[94;1m{:>3}\033[m)".format( fname, sample_sd, min_tile_sd )  )

            elif IsDegenerate:
              degenerate_image_count+=1
              if (DEBUG>2):
                 print ( "TILER: INFO:  \r\033[32Cskipping degenerate tile \r\033[65C\033[93m{:}\033[m \r\033[162Cwith \033[94;1m{:>3}\033[m unique values (minimum permitted is \033[94;1m{:>3}\033[m)".format( fname, unique_values, min_uniques )  )

            elif IsLowContrast:                                                                            # skip low information tiles
              low_contrast_tile_count       +=1
              if (DEBUG>2):
                print ( "TILER: INFO: \r\033[32Cskipping low contrast tile \r\033[65C\033[31m{:}\033[m \r\033[162Cwith greyscale range = \033[31;1m{:}\033[m (minimum permitted is \033[31;1m{:}\033[m)".format( fname, greyscale_range, greyness)  )                

            else:                  
              if not stain_norm =="NONE":                                                                  # then perform the selected stain normalization technique on the tile W

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


                tile = stain_normalization( norm_method, tile  )                                           # returns stain normalized version of the tile
              
              if (DEBUG>9):
                  print ( "TILER: INFO: saving   \r\033[65C\033[32m{:}\033[m, standard deviation = \033[32m{:>3.1f}\033[m".format( fname, sample_sd  ) )
              if (DEBUG>9):
                  print ( "TILER: INFO: saving   \r\033[65C\033[32m{:}\033[m with greyscale range = \033[32;1;4m{:}\033[m)".format( fname, greyscale_range) )

              if (DEBUG>9):
                print ( "TILER: INFO:               x = \033[1m{:}\033[m".format(x),             flush=True)
                print ( "TILER: INFO:               y = \033[1m{:}\033[m".format(y),             flush=True)  
                print ( "TILER: INFO:      tile_width = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:     tile_height = \033[1m{:}\033[m".format(tile_width),    flush=True)
                print ( "TILER: INFO:          fname  = \033[1m{:}\033[m".format( fname ) )

              tile.save(fname);                                                                            # save to the filename we made for this tile earlier              
              tiles_processed += 1
              
#              print ( "\033[s\033[{:};{:}f\033[32;1m{:}{:2d};{:>4d} \033[m\033[u".format( randint(1,68), int(1500/num_cpus)+7*my_thread, BB, my_thread+1, tiles_processed ), end="", flush=True )
              print ( f"\033[s\033[{tiles_processed//10};{int(1500/num_cpus)+7*my_thread}f\033[32;1m{BB}{my_thread+1:2d};{tiles_processed:>4d} \033[m\033[u", end="", flush=True )
    
  
  if (DEBUG>0):
    print ( f"\033[s\033[{my_thread+15};1f\
 \033[34mt=\033[1m{my_thread:>2d}\033[m\
 \033[34mc=\033[1m{tiles_considered_count:4d} \033[m\
 \033[34mok=\033[1m{tiles_processed:4d} \
 \033[1m{tiles_processed/tiles_considered_count *100:2.0f}%\033[m\
 \033[34mgr=\033[1m{low_contrast_tile_count:3d};\
 \033[1m{low_contrast_tile_count/tiles_considered_count *100:2.1f}%\033[m\
 \033[34mdeg=\033[1m{degenerate_image_count:3d} \
 \033[1m{degenerate_image_count/tiles_considered_count *100:2.1f}%\033[m\
 \033[34mbkg=\033[1m{background_image_count:4d} \
 \033[1m{background_image_count/tiles_considered_count *100:2.0f}% \033[m\
\033[u", flush=True, end="" ) 
  
  if (DEBUG>9):
    print('TILER: INFO: time taken to tile this SVS image: \033[1m{0:.2f}s\033[m'.format((time.time() - start)/60.0))

  if (DEBUG>9):
    print ( "TILER: INFO: about to display the \033[33;1m{:,}\033[m tiles".format    ( tiles_processed   ) )
    SUCCESS = display_processed_tiles( data_dir, DEBUG )

  return True

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

  return(1)
                  

# ------------------------------------------------------------------------------

def stain_normalization( norm_method, tile ):
  
  tile_rgb     = tile.convert('RGB')
  tile_rgb_npy = (np.array(tile_rgb))

  if (DEBUG>9):
    print ( "TILER:     INFO:  performing \033[35m{:}\033[m stain normalization on tile \033[35m{:}\033[m".format    ( stain_norm, fname  ), flush=True )


  tile_norm = norm_method.normalizer( tile_rgb_npy )                  #  ( path of source image )
  if (DEBUG>9):
    print ( "TILER:     INFO:  shape of stain normalized tile      = \033[36m{:}\033[m".format( tile_norm.shape ), flush=True )
  if (DEBUG>99):
    print ( "TILER:     INFO:  stain normalized tile               = \033[36m{:}\033[m".format( tile_norm       ), flush=True )

  tile_255 = tile_norm * 255
  if (DEBUG>99):
    np.set_printoptions(formatter={'float': lambda x: "{:3.2f}".format(x)})
    print ( "TILER:     INFO:  stain normalized tile shifted to 0-255   = \033[36m{:}\033[m".format( tile_255       ), flush=True )  

  tile_uint8 = np.uint8( tile_255 )
  if (DEBUG>99):
    np.set_printoptions(formatter={'int': lambda x: "{:>3d}".format(x)})
    print ( "TILER:     INFO:  stain normalized tile shifted to 0-255   = \033[36m{:}\033[m".format( tile_uint8       ), flush=True )   

  tile_norm_PIL = Image.fromarray( tile_uint8 )
  if (DEBUG>99):
    print ( "TILER:     INFO:  stain normalized tile as RGP PIL   = \033[36m{:}\033[m".format( tile_norm_PIL       ), flush=True )
    
  #tile_norm_PIL = Image.fromarray( np.uint8( np.random.rand(128,128,3) * 255 ) ) 
  tile = tile_norm_PIL.convert("RGB")
  if (DEBUG>99):
    print ( "TILER:     INFO:  stain normalized tile as RGP PIL   = \033[36m{:}\033[m".format( tile       ), flush=True )
    
  return tile

# ------------------------------------------------------------------------------
def check_background( tile,  points_to_sample, min_tile_sd ):

  tile_grey     = tile.convert('L')                                                            # make a greyscale copy of the image
  np_tile_grey  = np.array(tile_grey)
  
  sample       = [  np_tile_grey[randint(0,np_tile_grey.shape[0]-1), randint(0,np_tile_grey.shape[0]-1)] for x in range(0, points_to_sample) ]
  sample_mean  = np.mean(sample)
  sample_sd    = np.std (sample)
  sample_t     = ttest_1samp(sample, popmean=sample_mean)

  IsBackground=False
  if sample_sd<min_tile_sd:
    IsBackground=True

    if (DEBUG>2):
      print ( "\nTILER: INFO:  sample \033[94m\n{:}\033[m)".format   (    sample     ) )
      print ( "TILER: INFO:  len(sample) \033[94;1m{:}\033[m".format ( len(sample)   ) )
      print ( "TILER: INFO:  sample_mean \033[94;1m{:}\033[m".format (  sample_mean  ) )          
      print ( "TILER: INFO:  sample_sd \033[94;1m{:}\033[m".format   (   sample_sd   ) )     

  return IsBackground

# ------------------------------------------------------------------------------
def check_contrast( tile,  greyness ):

  # check greyscale range, as a proxy for useful information content
  tile_grey     = tile.convert('L')                                                                        # make a greyscale copy of the image
  greyscale_range  = np.max(np.array(tile_grey)) - np.min(np.array(tile_grey))                             # calculate the range of the greyscale copy
  GreyscaleRangeOk  = greyscale_range>greyness
  GreyscaleRangeBad = not GreyscaleRangeOk
  
  return GreyscaleRangeBad

# ------------------------------------------------------------------------------
def check_degeneracy( tile,  min_uniques ):

  # check number of unique values in the image, which we will use as a proxy to discover degenerate (articial) images
  unique_values = len(np.unique(tile )) 
  if (DEBUG>9):
    print ( "TILER: INFO:  number of unique values in this tile = \033[94;1;4m{:>3}\033[m) and minimum required is \033[94;1;4m{:>3}\033[m)".format( unique_values, min_uniques ) )
  IsDegenerate = unique_values<min_uniques
  
  return IsDegenerate
  
