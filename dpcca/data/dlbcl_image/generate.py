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

#from   data.gtexv6.config import GTExV6Config
from   data.dlbcl_image.config import GTExV6Config # NEW

np.set_printoptions(edgeitems=25)
np.set_printoptions(linewidth=240)

DEBUG                     = 1

BASE_DIR                  =  sys.argv[1]       # ~/biodata/dlbcl
TILE_SIZE                 =  int(sys.argv[2])  # 128 X 128 (assumed square)
MAX_ALLOWED_TILES_PER_SVS =  int(sys.argv[3])  
rna_file_name             =  sys.argv[4]       # "rna_scaled_estimate.npy"
tissue_class_file_name    =  sys.argv[5]       # "tissue_class.npy"

extension                 = "png"
NUMBER_OF_TILES           = 59*MAX_ALLOWED_TILES_PER_SVS    
NUMBER_OF_GENES           = 20531                           # must be EXACTLY the same as the number of genes in the 'scaled estimate' column of the rna csv file (.genes.results)

def main(cfg):

  images_new   = np.empty( (NUMBER_OF_TILES,  3, TILE_SIZE, TILE_SIZE), dtype=np.uint8  )                  # 6,039 * 128  * 1280  ~= 
  genes_new    = np.empty( (NUMBER_OF_TILES,  NUMBER_OF_GENES),         dtype=np.float64 )                 # 6,039 * 20,500       ~= 121MB
  tissues_new  = np.empty( (NUMBER_OF_TILES), dtype=np.int_)                                               # tissues_new holds tissue type class (integer), to be used as Truth labels by Torch in training 
  gnames_new   = np.empty( (NUMBER_OF_TILES), dtype=np.uint8)                                              # NOT USED
  fnames_new   = np.empty( (NUMBER_OF_TILES), dtype=np.uint8)                                              # tissue type name NOT USED
  
  i=-1    # gobal count of SVSs processed (directories stepped into) starting count at -1 because the top-level directory (patches), which contains no images, is also traversed
  j=0     # tiles processed per SVS (directory)
  k=0     # global count of tiles processed
  
  for dir_path, _, file_names in os.walk(BASE_DIR):

    print( "DLBCL_IMAGE: GENERATE: INFO: descending into folder \033[31;1m{:}{:}\033[m".format((len(dir_path.split(os.sep)) - 4) * '-', os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name

    j=0
    i+=1
    
    for file in file_names:
				
      if ( j<MAX_ALLOWED_TILES_PER_SVS ):
        image_file = os.path.join(dir_path, file)
        rna_file   = os.path.join(dir_path, rna_file_name)
        tissue_file   = os.path.join(dir_path, tissue_class_file_name)
        
        if DEBUG>1:
          if ( j%10==0 ):
            print ("DLBCL_IMAGE: GENERATE: INFO: dir_path   = {:}".format(dir_path))
        
        if ( file.endswith('.' + extension) & (not ( 'mask' in file ) ) & (not ( 'ized' in file ) )   ):   # because there are two other png files in each SVS folder besides the tile images

          if DEBUG>0:
            if ( j%10==0 ):
              print("DLBCL_IMAGE: GENERATE: INFO: about to process files {:} to {:} : for this SVS. Current file ({:})  = \033[33m{:}\033[m".format(j+1,j+10,j,image_file))
          
          #save_file = "{:}/TESTFILE".format(dir_path)
          #print ("save_file  = {:}".format(save_file))
          #shutil.copyfile( image_file, save_file )

          try:
            img = cv2.imread(image_file)
          except Exception as e:
            print ( "DLBCL_IMAGE: GENERATE: ERROR:	when opening image file \"{:}\"".format(e) )
          images_new [k,:] =  np.moveaxis(img, -1,0)
                   
          try:
            rna = np.load(rna_file)
            if DEBUG>1:
              print ( "DLBCL_IMAGE: rna.shape =  \"{:}\"".format( 	rna.shape) )
          except Exception as e:
            print ( "DLBCL_IMAGE: GENERATE: ERROR:	when opening rna file \"{:}\"".format(e) )
          genes_new  [k] =  rna[0]


          try:
            tissue_class = np.load(tissue_file)
            if DEBUG>1:
              print ( "DLBCL_IMAGE: tissue_class.shape =  \"{:}\"".format( 	tissue_class.shape) )
              print ( "DLBCL_IMAGE: tissue_class       =  \"{:}\"".format( 	tissue_class      ) )
              print ( "DLBCL_IMAGE: tissue_class[0]    =  \"{:}\"".format( 	tissue_class[0]   ) )
          except Exception as e:
            print ( "DLBCL_IMAGE: GENERATE: ERROR:	when opening tissue file \"{:}\"".format(e) )
          tissues_new[k] =  tissue_class[0]

          gnames_new [k]  =  443
          fnames_new [k]  =  738  

          if DEBUG>1:
            print ( "=" *180)
            print ( "DLBCL_IMAGE: GENERATE: INFO: tile {:} for this SVS:".format( j+1))
            print ( "DLBCL_IMAGE: GENERATE: INFO:   images_new[{:}].shape = {:}".format( k,  images_new[k].shape))
            print ( "DLBCL_IMAGE: GENERATE: INFO:       size in bytes = {:,}".format(images_new[k].size * images_new[k].itemsize))  
          if DEBUG>99:
            print ( "DLBCL_IMAGE: GENERATE: INFO:       value = \n{:}".format(images_new[k]))
  
          if DEBUG>1:
            print ( "DLBCL_IMAGE: GENERATE: INFO:   genes_new[{:}].shape = {:}".format( k,  genes_new[k].shape))
            print ( "DLBCL_IMAGE: GENERATE: INFO:       size in  bytes = {:,}".format(genes_new[k].size * genes_new[k].itemsize)) 
          if DEBUG>99:                                           
            print ( "DLBCL_IMAGE: GENERATE: INFO:       value = \n{:}".format(genes_new[k] ) )      
  
          if DEBUG>1:
            print ( "DLBCL_IMAGE: GENERATE: INFO:   tissues_new[{:}]".format( k ) )
            print ( "DLBCL_IMAGE: GENERATE: INFO:       size in  bytes = {:,}".format( tissues_new[k].size * tissues_new[k].itemsize ) ) 
            print ( "DLBCL_IMAGE: GENERATE: INFO:       value = {:}".format( tissues_new[k] ) )
  
          if DEBUG>99:
            print ( "DLBCL_IMAGE: GENERATE: INFO:   fnames_new[{:}]".format( k ) )
            print ( "DLBCL_IMAGE: GENERATE: INFO:       size in  bytes = {:,}".format( fnames_new[k].size * fnames_new[k].itemsize))
            print ( "DLBCL_IMAGE: GENERATE: INFO:       value = {:}".format( fnames_new[k] ) )
  
          if DEBUG>99:
            print ( "DLBCL_IMAGE: GENERATE: INFO:   gnames_new[{:}]".format( k ) )
            print ( "DLBCL_IMAGE: GENERATE: INFO:       size in  bytes = {:,}".format( gnames_new[k].size * gnames_new[k].itemsize))
            print ( "DLBCL_IMAGE: GENERATE: INFO:       value = {:}".format( gnames_new[k] ) )
         
          j+=1
          k+=1
          
        else:
          if DEBUG>1:
            print( "DLBCL_IMAGE: GENERATE: INFO: other file = \033[31m{:}\033[m".format(image_file)) 
        
        #time.sleep(.2)
        
  print ( "DLBCL_IMAGE: GENERATE: INFO: finished processing:")       
  print ( "DLBCL_IMAGE: GENERATE: INFO:    total number of SVSs  processed = \033[31m{:}\033[m".format(i))
  print ( "DLBCL_IMAGE: GENERATE: INFO:    user defined max tiles per SVS  = \033[31m{:}\033[m".format(MAX_ALLOWED_TILES_PER_SVS))
  print ( "DLBCL_IMAGE: GENERATE: INFO:    total number of tiles processed = \033[31m{:}\033[m".format(k))     

  print ( "DLBCL_IMAGE: GENERATE: INFO: (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(images_new)))
  print ( "DLBCL_IMAGE: GENERATE: INFO: (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(genes_new)))
  print ( "DLBCL_IMAGE: GENERATE: INFO: (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(fnames_new))) 
  print ( "DLBCL_IMAGE: GENERATE: INFO: (Numpy version of) tissues_new (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(fnames_new))) 
  print ( "DLBCL_IMAGE: GENERATE: INFO: (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof(gnames_new)))   
          
  # convert everything into torch style tensors
  images_new   = torch.Tensor( images_new  )
  print( "DLBCL_IMAGE: GENERATE: INFO: finished converting image data from numpy array to Torch tensor") 
  genes_new    = torch.Tensor( genes_new   )
  print( "DLBCL_IMAGE: GENERATE: INFO: finished converting rna   data from numpy array to Torch tensor")
  gnames_new   = torch.Tensor( gnames_new  )     
  tissues_new  = torch.Tensor( tissues_new).long()            # have to explicity cast as long as torch.Tensor does not automatically pick up type from the numpy array. 
  tissues_new.requires_grad_(False)                           # labels aren't allows grad
  fnames_new   = torch.Tensor( fnames_new  )  

  if DEBUG>0:
    print ( "DLBCL_IMAGE: GENERATE: INFO: shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "DLBCL_IMAGE: GENERATE: INFO: shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    print ( "DLBCL_IMAGE: GENERATE: INFO: shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))
    print ( "DLBCL_IMAGE: GENERATE: INFO: shape of (Torch version of) tissues_new.size = {:}".format(tissues_new.size() ))
    print ( "DLBCL_IMAGE: GENERATE: INFO: shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))
    
  if DEBUG>99: 
    print ( {
        'images':  images_new,
        'genes':   genes_new,
        'fnames':  fnames_new,
        'tissues': tissues_new,
        'gnames':  gnames_new        
    } )

  print( "DLBCL_IMAGE: GENERATE: INFO: now saving to Torch dictionary (this takes a little time)")
  
  torch.save({
      'images':  images_new,
      'genes':   genes_new,
      'fnames':  fnames_new,
      'tissues': tissues_new,
      'gnames':  gnames_new
  }, '%s/train.pth' % cfg.ROOT_DIR)

  print( "DLBCL_IMAGE: GENERATE: INFO: finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))   

  #print ("\n\033[31;1mtotal number of files processed and stored in numpy array = {:,}\033[m".format(j))

if __name__ == '__main__':
    cfg = GTExV6Config()
    main(cfg)    
