"""================================================================================================================================================

Routine to generate a dpcca TCGA-DBLC compatible python dictionary from already pre-processed TCGA image tiles and gene expression vectors PGD 191221

INPUT: (i)  root directory of image tiles ("patches" directory), which the code will traverse to locate png files for processing
       (ii) the routine expects to find a single rna expression file (*.results) in the same directory as the png file. 
            It will extract and create an rna expression vector from this file and store it in the dictionary with the same index as the image tile
            
================================================================================================================================================"""

import cv2
import os
import sys
import time
import shutil
import torch
import numpy as np
import pandas as pd

from   data.dlbcl_image.config import GTExV6Config

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

DEBUG                     = 1

BASE_DIR                  =  sys.argv[1]       # ~/git/pipeline
N_IMAGES                  =  int(sys.argv[2])
TILE_SIZE                 =  int(sys.argv[3])  # 128 X 128 (assumed square)
MAX_ALLOWED_TILES_PER_SVS =  int(sys.argv[4])  
rna_file_name             =  sys.argv[5]       # "rna.npy"
class_numpy_file_name     =  sys.argv[6]       # "class.npy" 
tile_extension            = "png"
  
NUMBER_OF_TILES           = N_IMAGES*MAX_ALLOWED_TILES_PER_SVS                                             # correct for sarc  
#NUMBER_OF_GENES          = 20531                                                                          # correct for dlbc / legacy portal version (.results)
NUMBER_OF_GENES           = 60482                                                                          # must be EXACTLY the same as the number of genes in the 'scaled estimate' column of the rna csv file ('mapping_file')


def main(cfg):

  images_new   = np.empty( ( NUMBER_OF_TILES,  3, TILE_SIZE, TILE_SIZE ), dtype=np.uint8   )                 # 6,039 * 128  * 1280  ~= 
  genes_new    = np.empty( ( NUMBER_OF_TILES,  NUMBER_OF_GENES         ), dtype=np.float64 )                 # 6,039 * 20,500       ~= 121MB
  labels_new   = np.empty( ( NUMBER_OF_TILES,                          ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
  gnames_new   = np.empty( ( NUMBER_OF_TILES                           ), dtype=np.uint8   )                 # was gene names       NOT USED
  fnames_new   = np.empty( ( NUMBER_OF_TILES                           ), dtype=np.uint8   )                 # was tissue type name NOT USED
  
  i=-1    # gobal count of SVSs processed (directories stepped into). Starting count is -1 because the top-level directory, which contains no images, is also traversed
  j=0     # tiles processed per SVS (directory)
  k=0     # global count of tiles processed
  
  for dir_path, _, file_names in os.walk(BASE_DIR):

    j=0
    i+=1

    print( "GENERATE_IMAGE:        descending into folder \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-',   i, os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name

    for file in file_names:

      if ( j<MAX_ALLOWED_TILES_PER_SVS ):
        image_file = os.path.join(dir_path, file)
        rna_file      = os.path.join(dir_path, rna_file_name)
        label_file   = os.path.join(dir_path, class_numpy_file_name)
        
        if DEBUG>1:
          if ( j%10==0 ):
            print ("GENERATE_IMAGE:        INFO: dir_path   = {:}".format(dir_path))
        
        if ( file.endswith('.' + tile_extension) & (not ( 'mask' in file ) ) & (not ( 'ized' in file ) )   ):   # because there are two other png files in each SVS folder besides the tile images

          if DEBUG>0:
            if ( j%50==0 ):
              print("GENERATE_IMAGE:        INFO: about to process files {0:4d} to {1:4d} : for this SVS. Current file ({2:4d})  = \033[33m{3:s}\033[m".format( j+1, j+50,j,image_file))

          try:
            img = cv2.imread(image_file)
          except Exception as e:
            print ( "GENERATE_IMAGE:        ERROR: when opening this image file -- skipping \"{:}\"".format(e) )
          images_new [k,:] =  np.moveaxis(img, -1,0)
                   
          try:
            rna = np.load(rna_file)
            if DEBUG>1:
              print ( "DLBCL_IMAGE: rna.shape =  \"{:}\"".format( 	rna.shape) )
          except Exception as e:
            print ( "GENERATE_IMAGE:        ERROR: when opening this rna file -- skipping \"{:}\"".format(e) )
          genes_new  [k] =  rna[0]

          try:
            label = np.load(label_file)
            if DEBUG>1:
              print ( "DLBCL_IMAGE: label.shape =  \"{:}\"".format(  label.shape) )
              print ( "DLBCL_IMAGE: label       =  \"{:}\"".format(  label      ) )
              print ( "DLBCL_IMAGE: label[0]    =  \"{:}\"".format(  label[0]   ) )
          except Exception as e:
            print ( "GENERATE_IMAGE:        ERROR: when opening this label file -- skipping\"{:}\"".format(e) )
            
          labels_new[k] =  label[0]

          gnames_new [k]  =  443                                                                           # Any old number. We don't currently use these
          fnames_new [k]  =  738                                                                           # Any old number. We don't currently use these  

          if DEBUG>9:
            print ( "=" *180)
            print ( "GENERATE_IMAGE:        INFO: tile {:} for this SVS:".format( j+1))
            print ( "GENERATE_IMAGE:        INFO:   images_new[{:}].shape = {:}".format( k,  images_new[k].shape))
            print ( "GENERATE_IMAGE:        INFO:       size in bytes = {:,}".format(images_new[k].size * images_new[k].itemsize))  
          if DEBUG>99:
            print ( "GENERATE_IMAGE:        INFO:       value = \n{:}".format(images_new[k]))
  
          if DEBUG>9:
            print ( "GENERATE_IMAGE:        INFO:   genes_new[{:}].shape = {:}".format( k,  genes_new[k].shape))
            print ( "GENERATE_IMAGE:        INFO:       size in  bytes = {:,}".format(genes_new[k].size * genes_new[k].itemsize)) 
          if DEBUG>99:                                           
            print ( "GENERATE_IMAGE:        INFO:       value = \n{:}".format(genes_new[k] ) )      
  
          if DEBUG>9:
            print ( "GENERATE_IMAGE:        INFO:   labels_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE:        INFO:       size in  bytes = {:,}".format( labels_new[k].size * labels_new[k].itemsize ) ) 
            print ( "GENERATE_IMAGE:        INFO:       value = {:}".format( labels_new[k] ) )
  
          if DEBUG>99:
            print ( "GENERATE_IMAGE:        INFO:   fnames_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE:        INFO:       size in  bytes = {:,}".format( fnames_new[k].size * fnames_new[k].itemsize))
            print ( "GENERATE_IMAGE:        INFO:       value = {:}".format( fnames_new[k] ) )
  
          if DEBUG>99:
            print ( "GENERATE_IMAGE:        INFO:   gnames_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE:        INFO:       size in  bytes = {:,}".format( gnames_new[k].size * gnames_new[k].itemsize))
            print ( "GENERATE_IMAGE:        INFO:       value = {:}".format( gnames_new[k] ) )
         
          j+=1
          k+=1
          
        else:
          if DEBUG>1:
            print( "GENERATE_IMAGE:        INFO: other file = \033[31m{:}\033[m".format( image_file ) ) 
        
  print ( "GENERATE_IMAGE:        INFO: finished processing:")       
  print ( "GENERATE_IMAGE:        INFO:    total number of SVSs  processed = \033[31m{:}\033[m".format(i))
  print ( "GENERATE_IMAGE:        INFO:    user defined max tiles per SVS  = \033[31m{:}\033[m".format(MAX_ALLOWED_TILES_PER_SVS))
  print ( "GENERATE_IMAGE:        INFO:    total number of tiles processed = \033[31m{:}\033[m".format(k))     

  print ( "GENERATE_IMAGE:        INFO: (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new )))
  print ( "GENERATE_IMAGE:        INFO: (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( genes_new  )))
  print ( "GENERATE_IMAGE:        INFO: (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 
  print ( "GENERATE_IMAGE:        INFO: (Numpy version of) labels_new (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 
  print ( "GENERATE_IMAGE:        INFO: (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new )))   
          
  # convert everything into Torch style tensors
  images_new   = torch.Tensor( images_new  )
  print( "GENERATE_IMAGE:        INFO: finished converting image data from numpy array to Torch tensor") 
  genes_new    = torch.Tensor( genes_new   )
  print( "GENERATE_IMAGE:        INFO: finished converting rna   data from numpy array to Torch tensor")
  gnames_new   = torch.Tensor( gnames_new  )     
  labels_new  = torch.Tensor( labels_new).long()                                                         # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  print( "GENERATE_IMAGE:        INFO: finished converting labels from numpy array to Torch tensor")
  labels_new.requires_grad_( False )                                                                      # labels aren't allows gradients
  fnames_new   = torch.Tensor( fnames_new  )

  if DEBUG>0:
    print ( "GENERATE_IMAGE:        INFO: shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "GENERATE_IMAGE:        INFO: shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    print ( "GENERATE_IMAGE:        INFO: shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))
    print ( "GENERATE_IMAGE:        INFO: shape of (Torch version of) labels_new.size = {:}".format(labels_new.size() ))
    print ( "GENERATE_IMAGE:        INFO: shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))
    
  if DEBUG>99: 
    print ( {
        'images':  images_new,
        'genes':   genes_new,
        'fnames':  fnames_new,
        'tissues': labels_new,
        'gnames':  gnames_new        
    } )

  print( "GENERATE_IMAGE:        INFO: now saving to Torch dictionary (this takes a little time)")
  
  torch.save({
      'images':  images_new,
      'genes':   genes_new,
      'fnames':  fnames_new,
      'tissues': labels_new,
      'gnames':  gnames_new
  }, '%s/train.pth' % cfg.ROOT_DIR)

  print( "GENERATE_IMAGE:        INFO: finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))   

  #print ("\n\033[31;1mtotal number of files processed and stored in numpy array = {:,}\033[m".format(j))

if __name__ == '__main__':

    cfg = GTExV6Config( 0, 0 )

    main(cfg)    
