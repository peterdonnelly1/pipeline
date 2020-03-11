"""================================================================================================================================================

Routine to generate a dpcca TCGA-DBLC compatible python dictionary from already pre-processed TCGA image tiles and gene expression vectors PGD 191221++

INPUT: (image_count)  the code will traverse data_dir to locate png files for processing
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

from  data.dlbcl_image.config import GTExV6Config

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

DEBUG=1



def generate_image( args, n_samples ):

  data_dir              = args.base_dir
  tile_size             = args.tile_size
  rna_file_name         = args.rna_file_name
  class_numpy_file_name = args.class_numpy_file_name
  n_tiles               = args.n_tiles             # tiles per image desired
  n_genes               = args.n_genes             # must be EXACTLY the same as the number of genes in the 'scaled estimate' column of the rna csv file ('mapping_file')

  print( "GENERATE_IMAGE: INFO:    at generate_image(): \
 data_dir=\033[36;1m{:}\033[m,\
 tile_size=\033[36;1m{:}\033[m,\
 rna_file_name=\033[36;1m{:}\033[m,\
 class_numpy_file_name=\033[36;1m{:}\033[m,\
 n_tiles=\033[36;1m{:}\033[m,\
 n_genes=\033[36;1m{:}\033[m"\
.format( args.data_dir, args.tile_size, args.rna_file_name, args.class_numpy_file_name, args.n_tiles, args.n_genes ), flush=True )

  data_dir              = args.data_dir
  tile_size             = args.tile_size
  rna_file_name         = args.rna_file_name
  class_numpy_file_name = args.class_numpy_file_name
  n_tiles               = args.n_tiles
  n_genes               = args.n_genes
 
  total_tiles           = n_samples*n_tiles
  tile_extension        = "png"

  cfg = GTExV6Config( 0,0 )
  
  images_new   = np.empty( ( total_tiles,  3, tile_size, tile_size ), dtype=np.uint8   )                 # 6,039 * 128  * 1280  ~= 
  genes_new    = np.empty( ( total_tiles,  n_genes                 ), dtype=np.float64 )                 # 6,039 * 20,500       ~= 121MB
  labels_new   = np.empty( ( total_tiles,                          ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
  gnames_new   = np.empty( ( total_tiles                           ), dtype=np.uint8   )                 # was gene names       NOT USED
  fnames_new   = np.empty( ( total_tiles                           ), dtype=np.uint8   )                 # was tissue type name NOT USED
  
  image_count=0     # gobal count of images processed (directories stepped into). Care! The top-level directory, which contains no images, is also traversed
  #tile_count=0     # tiles processed per images (directory)
  k=0               # global count of tiles processed
  
  for dir_path, _, file_names in os.walk( data_dir ):
  
    tile_count=0   # tiles processed per images (directory)
  
    print( "GENERATE_IMAGE: INFO:     descending into folder \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-',   image_count, os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name
    
    for file in file_names:
    
      if ( tile_count<n_tiles ):
        image_file = os.path.join(dir_path, file)
        rna_file   = os.path.join(dir_path, rna_file_name)
        label_file = os.path.join(dir_path, class_numpy_file_name)
      
        if DEBUG>1:
          if ( tile_count%10==0 ):
            print ("GENERATE_IMAGE: INFO:       dir_path   = {:}".format(dir_path))
      
        if ( file.endswith('.' + tile_extension) & (not ( 'mask' in file ) ) & (not ( 'ized' in file ) )   ):   # because there are two other png files in each image folder besides the tile images
      
          if DEBUG>99999999:
            if ( tile_count%50==0 ):
              print("GENERATE_IMAGE: INFO:       about to process files {0:4d} to {1:4d} : for this image. Current file ({2:4d})  = \033[33m{3:s}\033[m".format( tile_count+1, tile_count+50,tile_count, image_file))
      
          try:
            img = cv2.imread(image_file)
          except Exception as e:
            print ( "GENERATE_IMAGE: ERROR:    when opening this image file -- skipping \"{:}\"".format(e) )
            images_new [k,:] =  np.moveaxis(img, -1,0)
               
          try:
            rna = np.load(rna_file)
            if DEBUG>1:
              print ( "GENERATE_IMAGE: INFO:       rna.shape =  \"{:}\"".format(   rna.shape) )
          except Exception as e:
            print ( "GENERATE_IMAGE: ERROR:    when opening this rna file -- skipping \"{:}\"".format(e) )
            genes_new  [k] =  rna[0]
      
          try:
            label = np.load(label_file)
            if DEBUG>1:
              print ( "GENERATE_IMAGE: INFO:       label.shape =  \"{:}\"".format(  label.shape) )
              print ( "GENERATE_IMAGE: INFO:       label       =  \"{:}\"".format(  label      ) )
              print ( "GENERATE_IMAGE: INFO:       label[0]    =  \"{:}\"".format(  label[0]   ) )
          except Exception as e:
            print ( "GENERATE_IMAGE: ERROR:    when opening this label file -- skipping\"{:}\"".format(e) )
            
          labels_new[k] =  label[0]
      
          gnames_new [k]  =  443                                                                           # Any old number. We don't currently use these
          fnames_new [k]  =  738                                                                           # Any old number. We don't currently use these  
      
          if DEBUG>9:
            print ( "=" *180)
            print ( "GENERATE_IMAGE: INFO:       tile {:} for this image:".format( tile_count+1))
            print ( "GENERATE_IMAGE: INFO:         images_new[{:}].shape = {:}".format( k,  images_new[k].shape))
            print ( "GENERATE_IMAGE: INFO:             size in bytes = {:,}".format(images_new[k].size * images_new[k].itemsize))  
            if DEBUG>99:
              print ( "GENERATE_IMAGE: INFO:             value = \n{:}".format(images_new[k]))
      
          if DEBUG>9:
            print ( "GENERATE_IMAGE: INFO:         genes_new[{:}].shape = {:}".format( k,  genes_new[k].shape))
            print ( "GENERATE_IMAGE: INFO:             size in  bytes = {:,}".format(genes_new[k].size * genes_new[k].itemsize)) 
            if DEBUG>99:                                           
              print ( "GENERATE_IMAGE: INFO:             value = \n{:}".format(genes_new[k] ) )      
      
          if DEBUG>9:
            print ( "GENERATE_IMAGE: INFO:       labels_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE: INFO:           size in  bytes = {:,}".format( labels_new[k].size * labels_new[k].itemsize ) ) 
            print ( "GENERATE_IMAGE: INFO:           value = {:}".format( labels_new[k] ) )
      
          if DEBUG>99:
            print ( "GENERATE_IMAGE: INFO:         fnames_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE: INFO:             size in  bytes = {:,}".format( fnames_new[k].size * fnames_new[k].itemsize))
            print ( "GENERATE_IMAGE: INFO:             value = {:}".format( fnames_new[k] ) )
      
          if DEBUG>99:
            print ( "GENERATE_IMAGE: INFO:         gnames_new[{:}]".format( k ) )
            print ( "GENERATE_IMAGE: INFO:             size in  bytes = {:,}".format( gnames_new[k].size * gnames_new[k].itemsize))
            print ( "GENERATE_IMAGE: INFO:             value = {:}".format( gnames_new[k] ) )
         
          tile_count+=1
          k+=1
          if DEBUG>0:
            print ( "GENERATE_IMAGE: INFO:       tile_count = {0:>3d};  number required = {1:>3d};  global tiles processed = {2:>6d}".format( tile_count,  n_tiles, k ) )
        
        else:
          if DEBUG>1:
            print( "GENERATE_IMAGE: INFO:     other file = \033[31m{:}\033[m".format( image_file ) ) 

      if DEBUG>9:
        print( "GENERATE_IMAGE: INFO:     image_count, n_samples = \033[31m{:}, {:}\033[m".format( image_count, n_samples) )
      
    image_count+=1
    if image_count>n_samples:
      break


  print ( "GENERATE_IMAGE: INFO:     finished processing:")       
  print ( "GENERATE_IMAGE: INFO:        total number of images  processed = \033[31m{:}\033[m".format(image_count-1))
  print ( "GENERATE_IMAGE: INFO:        user defined max tiles per image  = \033[31m{:}\033[m".format(n_tiles))
  print ( "GENERATE_IMAGE: INFO:        total number of tiles processed   = \033[31m{:}\033[m".format(k))     
  
  print ( "GENERATE_IMAGE: INFO:     (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new )))
  print ( "GENERATE_IMAGE: INFO:     (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( genes_new  )))
  print ( "GENERATE_IMAGE: INFO:     (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 
  print ( "GENERATE_IMAGE: INFO:     (Numpy version of) labels_new (dummy data) ---------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 
  print ( "GENERATE_IMAGE: INFO:     (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new )))   

          
  # convert everything into Torch style tensors
  images_new   = torch.Tensor( images_new  )
  print( "GENERATE_IMAGE: INFO:     finished converting image data from numpy array to Torch tensor") 
  genes_new    = torch.Tensor( genes_new   )
  print( "GENERATE_IMAGE: INFO:     finished converting rna   data from numpy array to Torch tensor")
  gnames_new   = torch.Tensor( gnames_new  )     
  labels_new  = torch.Tensor( labels_new).long()                                                         # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  print( "GENERATE_IMAGE: INFO:     finished converting labels from numpy array to Torch tensor")
  labels_new.requires_grad_( False )                                                                      # labels aren't allows gradients
  fnames_new   = torch.Tensor( fnames_new  )
  
  if DEBUG>0:
    print ( "GENERATE_IMAGE: INFO:     shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "GENERATE_IMAGE: INFO:     shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    print ( "GENERATE_IMAGE: INFO:     shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))
    print ( "GENERATE_IMAGE: INFO:     shape of (Torch version of) labels_new.size  = {:}".format(labels_new.size() ))
    print ( "GENERATE_IMAGE: INFO:     shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))
  
  if DEBUG>99: 
    print ( {
      'images':  images_new,
      'genes':   genes_new,
      'fnames':  fnames_new,
      'tissues': labels_new,
      'gnames':  gnames_new        
    } )
    
  print( "GENERATE_IMAGE: INFO:     now saving to Torch dictionary (this takes a little time)")
  
  torch.save({
    'images':  images_new,
    'genes':   genes_new,
    'fnames':  fnames_new,
    'tissues': labels_new,
    'gnames':  gnames_new
  }, '%s/train.pth' % cfg.ROOT_DIR)
  
  print( "GENERATE_IMAGE: INFO:     finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))   
  
  #print ("\n\033[31;1mtotal number of files processed and stored in numpy array = {:,}\033[m".format(tile_count))

