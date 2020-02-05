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

from   data.gtexv6.config import GTExV6Config

np.set_printoptions(edgeitems=25)
np.set_printoptions(linewidth=240)

DEBUG                     = 1

BASE_DIR                  =  sys.argv[1]       # ~/git/pipeline
TILE_SIZE                 =  int(sys.argv[2])  # 128 X 128 (assumed square)
MAX_ALLOWED_TILES_PER_SVS =  int(sys.argv[3])  
rna_file_name             =  sys.argv[4]       # "rna_scaled_estimate.npy"
class_numpy_file_name     =  sys.argv[5]       # "class.npy"      
tile_extension                 = "png"
  
NUMBER_OF_TILES           = 59*MAX_ALLOWED_TILES_PER_SVS     # correct for dlbc  
#NUMBER_OF_TILES           = 67*MAX_ALLOWED_TILES_PER_SVS     # correct for eye  
#NUMBER_OF_GENES           = 20531                           # correct for dlbc / legacy portal version (.results)
NUMBER_OF_GENES           = 60482                            # must be EXACTLY the same as the number of genes in the 'scaled estimate' column of the rna csv file (.genes.results)

def main(cfg):

  images_new   = np.empty( (NUMBER_OF_TILES,  3, TILE_SIZE, TILE_SIZE), dtype=np.uint8  )                  # 6,039 * 128  * 1280  ~= 
  genes_new    = np.empty( (NUMBER_OF_TILES,  NUMBER_OF_GENES),         dtype=np.float64 )                 # 6,039 * 20,500       ~= 121MB
  tissues_new  = np.empty( (NUMBER_OF_TILES), dtype=np.uint8)                                              # tissues_new holds tissue type class (integer), to be used as Truth labels by Torch in training 
  gnames_new   = np.empty( (NUMBER_OF_TILES), dtype=np.uint8)                                              # NOT USED
  fnames_new   = np.empty( (NUMBER_OF_TILES), dtype=np.uint8)                                              # tissue type name NOT USED
  
  i=-1    # gobal count of SVSs processed (directories stepped into) starting count at -1 because the top-level directory (patches), which contains no images, is also traversed
  j=0     # tiles processed per SVS (directory)
  k=0     # global count of tiles processed
  
  for dir_path, _, file_names in os.walk(BASE_DIR):

    j=0
    i+=1

    print( "GTEXV6: GENERATE: INFO: descending into folder \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-',   i, os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name

    
    for file in file_names:
				
      if ( j<MAX_ALLOWED_TILES_PER_SVS ):
        image_file = os.path.join(dir_path, file)
        rna_file   = os.path.join(dir_path, rna_file_name)
        
        if DEBUG>1:
          if ( j%10==0 ):
            print ("GTEXV6: GENERATE: INFO: dir_path   = {:}".format(dir_path))
        
        if ( file.endswith('.' + tile_extension) & (not ( 'mask' in file ) ) & (not ( 'ized' in file ) )   ):   # because there are two other png files in each SVS folder besides the tile images

          if DEBUG>0:
            if ( j%50==0 ):
              print("GTEXV6: GENERATE: INFO: about to process files {0:4d} to {1:4d} : for this SVS. Current file ({2:4d})  = \033[33m{3:s}\033[m".format( j+1, j+50,j,image_file))
          
          #save_file = "{:}/TESTFILE".format(dir_path)
          #print ("save_file  = {:}".format(save_file))
          #shutil.copyfile( image_file, save_file )

          try:
            img = cv2.imread(image_file)
          except Exception as e:
            print ( "GTEXV6: GENERATE: ERROR:	when opening image file \"{:}\"".format(e) )
          images_new [k,:] =  np.moveaxis(img, -1,0)
                   
          try:
            rna = np.load(rna_file)
            if DEBUG>1:
              print ( "GTEXV6: rna.shape =  \"{:}\"".format( 	rna.shape) )
          except Exception as e:
            print ( "GTEXV6: GENERATE: ERROR:	when opening rna file \"{:}\"".format(e) )
          genes_new  [k] =  rna[:,0]
          
          tissues_new[k]  =  7                                                                             # all the same kind of tissue for dblc
          gnames_new [k]  =  44
          fnames_new [k]  =  73  

          if DEBUG>1:
            print ( "=" *180)
            print ( "GTEXV6: GENERATE: INFO: tile {:} for this SVS:".format( j+1))
            print ( "GTEXV6: GENERATE: INFO:   images_new[{:}].shape = {:}".format( k,  images_new[k].shape))
            print ( "GTEXV6: GENERATE: INFO:       size in bytes = {:,}".format(images_new[k].size * images_new[k].itemsize))  
          if DEBUG>99:
            print ( "GTEXV6: GENERATE: INFO:       value = \n{:}".format(images_new[k]))
  
          if DEBUG>1:
            print ( "GTEXV6: GENERATE: INFO:   genes_new[{:}].shape = {:}".format( k,  genes_new[k].shape))
            print ( "GTEXV6: GENERATE: INFO:       size in  bytes = {:,}".format(genes_new[k].size * genes_new[k].itemsize)) 
          if DEBUG>99:                                           
            print ( "GTEXV6: GENERATE: INFO:       value = \n{:}".format(genes_new[k] ) )      
  
          if DEBUG>1:
            print ( "GTEXV6: GENERATE: INFO:   tissues_new[{:}]".format( k ) )
            print ( "GTEXV6: GENERATE: INFO:       size in  bytes = {:,}".format( tissues_new[k].size * tissues_new[k].itemsize ) ) 
            print ( "GTEXV6: GENERATE: INFO:       value = {:}".format( tissues_new[k] ) )
  
          if DEBUG>99:
            print ( "GTEXV6: GENERATE: INFO:   fnames_new[{:}]".format( k ) )
            print ( "GTEXV6: GENERATE: INFO:       size in  bytes = {:,}".format( fnames_new[k].size * fnames_new[k].itemsize))
            print ( "GTEXV6: GENERATE: INFO:       value = {:}".format( fnames_new[k] ) )
  
          if DEBUG>99:
            print ( "GTEXV6: GENERATE: INFO:   gnames_new[{:}]".format( k ) )
            print ( "GTEXV6: GENERATE: INFO:       size in  bytes = {:,}".format( gnames_new[k].size * gnames_new[k].itemsize))
            print ( "GTEXV6: GENERATE: INFO:       value = {:}".format( gnames_new[k] ) )
         
          j+=1
          k+=1
          
        else:
          if DEBUG>1:
            print( "GTEXV6: GENERATE: INFO: other file = \033[31m{:}\033[m".format(image_file)) 
        
        #time.sleep(.2)
        
  print ( "GTEXV6: GENERATE: INFO: finished processing:")       
  print ( "GTEXV6: GENERATE: INFO:    total number of SVSs  processed = \033[31m{:}\033[m".format(i))
  print ( "GTEXV6: GENERATE: INFO:    user defined max tiles per SVS  = \033[31m{:}\033[m".format(MAX_ALLOWED_TILES_PER_SVS))
  print ( "GTEXV6: GENERATE: INFO:    total number of tiles processed = \033[31m{:}\033[m".format(k))     

  print ( "GTEXV6: GENERATE: INFO: (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(images_new)))
  print ( "GTEXV6: GENERATE: INFO: (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(genes_new)))
  print ( "GTEXV6: GENERATE: INFO: (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(fnames_new))) 
  print ( "GTEXV6: GENERATE: INFO: (Numpy version of) tissues_new (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(fnames_new))) 
  print ( "GTEXV6: GENERATE: INFO: (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(gnames_new)))   
          
  # convert everything into torch style tensors
  images_new   = torch.Tensor( images_new  )
  print( "GTEXV6: GENERATE: INFO: finished converting image data from numpy array to Torch tensor") 
  genes_new    = torch.Tensor( genes_new   )
  print( "GTEXV6: GENERATE: INFO: finished converting rna   data from numpy array to Torch tensor")
  gnames_new   = torch.Tensor( gnames_new  )     
  tissues_new  = torch.Tensor( tissues_new )
  fnames_new   = torch.Tensor( fnames_new  )

  if DEBUG>0:
    print ( "GTEXV6: GENERATE: INFO: shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "GTEXV6: GENERATE: INFO: shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    print ( "GTEXV6: GENERATE: INFO: shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))
    print ( "GTEXV6: GENERATE: INFO: shape of (Torch version of) tissues_new.size = {:}".format(tissues_new.size() ))
    print ( "GTEXV6: GENERATE: INFO: shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))
    
  if DEBUG>9: 
    print ( {
        'images':  images_new,
        'genes':   genes_new,
        'fnames':  fnames_new,
        'tissues': tissues_new,
        'gnames':  gnames_new        
    } )

  print( "GTEXV6: GENERATE: INFO: now saving to Torch dictionary (this takes a little time)")
  
  torch.save({
      'images':  images_new,
      'genes':   genes_new,
      'fnames':  fnames_new,
      'tissues': tissues_new,
      'gnames':  gnames_new
  }, '%s/train.pth' % cfg.ROOT_DIR)

  print( "GTEXV6: GENERATE: INFO: finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))   

  #print ("\n\033[31;1mtotal number of files processed and stored in numpy array = {:,}\033[m".format(j))

if __name__ == '__main__':
    cfg = GTExV6Config()
    main(cfg)    
