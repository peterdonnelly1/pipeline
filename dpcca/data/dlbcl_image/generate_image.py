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


def generate_image( args, n_samples, n_tiles, n_genes, gene_data_norm ):

  # DON'T USE args.n_samples or args.n_tiles or args.gene_data_norm since they are the complete, job-level lists. Here we are just using one of each, passed in as the parameters above
  data_dir                = args.data_dir
  input_mode              = args.input_mode                                                                  # suppress generation of RNA related data
  tile_size               = args.tile_size
  rna_file_name           = args.rna_file_name
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  class_numpy_file_name   = args.class_numpy_file_name

  if input_mode=='image':
    print( f"GENERATE:       INFO:      GENERATE:      (): input_mode is '\033[31;1m\033[3m{input_mode}\033[m' GENERATION OF RNA DATA WILL BE SUPPRESSED!" )  


  print( "GENERATE:       INFO:      at GENERATE:      (): \
 data_dir=\033[36;1m{:}\033[m,\
 n_samples=\033[36;1m{:}\033[m,\
 n_tiles=\033[36;1m{:}\033[m,\
 tile_size=\033[36;1m{:}\033[m,\
 rna_file_name=\033[36;1m{:}\033[m,\
 class_numpy_file_name=\033[36;1m{:}\033[m,\
 n_tiles=\033[36;1m{:}\033[m,\
 n_genes=\033[36;1m{:}\033[m"\
.format( data_dir, n_samples, n_tiles, tile_size, rna_file_name, class_numpy_file_name, n_tiles, n_genes ), flush=True )
 
  total_tiles           = n_samples*n_tiles
  tile_extension        = "png"

  if DEBUG>0:
    print ( f"GENERATE:       INFO:            n_samples   = {n_samples}" )
    if input_mode=='image':  
      print ( f"GENERATE:       INFO:            n_tiles     = {n_tiles}" )      
      print ( f"GENERATE:       INFO:            total_tiles = {total_tiles}" )  
    if input_mode=='image':  
      print ( f"GENERATE:       INFO:            n_genes     = {n_genes}" )      

  cfg = GTExV6Config( 0,0 )

  if input_mode=='image':
    images_new   = np.empty( ( total_tiles,  3, tile_size, tile_size ), dtype=np.uint8   )                 #
    fnames_new   = np.empty( ( total_tiles                           ), dtype=np.uint8   )                 # was tissue type name NOT USED
    labels_new   = np.empty( ( total_tiles,                          ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
    tiles_processed        =  0     # tiles processed per SVS image (directory)
    global_tiles_processed =  0     # global count of tiles processed 
  if input_mode=='rna':
    genes_new    = np.empty( ( n_samples, 1, n_genes                 ), dtype=np.float64 )                 #
    gnames_new   = np.empty( ( n_samples                             ), dtype=np.uint8   )                 # was gene names       NOT USED
    labels_new   = np.empty( ( n_samples,                            ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
    global_genes_processed =  0     # global count of genes processed
  
  samples_processed      = -1     # gobal count of samples processed (directories stepped into). Starting count is -1 because the top-level directory, which contains no images, is also traversed
      
  for dir_path, dirs, file_names in os.walk( data_dir ):

    tiles_processed         = 0
    samples_processed      += 1

    if samples_processed>n_samples:
      break

    print( "GENERATE:       INFO:      descending into folder \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-',   samples_processed, os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name

    for file in sorted(file_names):                                                                                # examine every file in the current directory

      if DEBUG>999:  
        print( f"GENERATE:       INFO:                     rna = \n\033[31m{file_names}\033[m" )
            
      if DEBUG>999:  
        print( f"\nGENERATE:       INFO:                     rna = \n\033[31m{sorted(file_names)}\033[m" )

      if input_mode=='image':

        if ( tiles_processed<n_tiles ):                                                                    # while we have less than the requested number of tiles for this SVS image (directory)
          
          image_file    = os.path.join(dir_path, file)
          label_file    = os.path.join(dir_path, class_numpy_file_name)
          
          if DEBUG>1:
            if ( tiles_processed%10==0 ):
              print ("GENERATE:       INFO:          dir_path   = {:}".format(dir_path))
          
          if ( file.endswith('.' + tile_extension) & (not ( 'mask' in file ) ) & (not ( 'ized' in file ) )   ):   # because there may be other png files in each image folder besides the tile image files
  
            if DEBUG>0:
              if ( tiles_processed%50==0 ):
                print("GENERATE:       INFO:          about to process files {0:4d} to {1:4d} : for this image. Current file ({2:4d})  = \033[33m{3:s}\033[m".format( tiles_processed+1, tiles_processed+50, tiles_processed, image_file))
  
            try:
              img = cv2.imread(image_file)
            except Exception as e:
              print ( "GENERATE:      :        ERROR: when opening this image file -- skipping \"{:}\"".format(e) )

            images_new [global_tiles_processed,:] =  np.moveaxis(img, -1,0)                                 # add it to the images array
  
            try:                                                                                            # every tile has an associated label - the same label for every tile image in the directory
              label = np.load(label_file)
              if DEBUG>99:
                print ( "GENERATE:      : label.shape =  \"{:}\"".format(  label.shape) )
                print ( "GENERATE:      : label       =  \"{:}\"".format(  label      ) )
              if DEBUG>999:
                print ( f"{label[0]},", end='', flush=True )
            except Exception as e:
              print ( "GENERATE:      :        ERROR: when opening this label file -- skipping\"{:}\"".format(e) )
              
            labels_new[global_tiles_processed] =  label[0]                                                 # add it to the labels array

            if DEBUG>99:
              print ( f"{label[0]},", flush=True, end="" )

            fnames_new [global_tiles_processed]  =  738                                                    # Any old number. We don't currently use these 
  
            if DEBUG>9:
              print ( "=" *180)
              print ( "GENERATE:       INFO:          tile {:} for this image:".format( tiles_processed+1))
              print ( "GENERATE:       INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
              print ( "GENERATE:       INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
            if DEBUG>99:
              print ( "GENERATE:       INFO:                value = \n{:}".format(images_new[global_tiles_processed]))
    
            if DEBUG>9:
              print ( "GENERATE:       INFO:            labels_new[{:}]".format( global_tiles_processed ) )
              print ( "GENERATE:       INFO:                size in  bytes = {:,}".format( labels_new[global_tiles_processed].size * labels_new[global_tiles_processed].itemsize ) ) 
              print ( "GENERATE:       INFO:                value = {:}".format( labels_new[global_tiles_processed] ) )
    
            if DEBUG>99:
              print ( "GENERATE:       INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
              print ( "GENERATE:       INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
              print ( "GENERATE:       INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
             
            tiles_processed+=1
            global_tiles_processed+=1
            
          else:
            if DEBUG>1:
              print( "GENERATE:       INFO:          other file = \033[31m{:}\033[m".format( image_file ) ) 
  
              
      elif input_mode=='rna':

        if not (dir_path==data_dir):                                                                       # the top level directory (dataset) has be skipped because it only contains sub-directories, not data

          if ( file.endswith(rna_file_reduced_suffix) ):
        
            rna_file      = os.path.join(dir_path, rna_file_name)
            label_file    = os.path.join(dir_path, class_numpy_file_name)
            
            try:
              rna = np.load( rna_file )
              if DEBUG>1:
                print ( "\nGENERATE:      : rna.shape =  \"{:}\"".format( rna.shape) )
            except Exception as e:
              print ( f"\033[31mGENERATE:       FATAL: {e} ... halting now [118]\033[m" )
              sys.exit(0)
                                                                          # remove row zero, which just holds the size of the file
            if DEBUG>9:  
              print( f"GENERATE:       INFO:                     rna = \n\033[31m{rna}\033[m" )              
            normalized_rna = ( rna - np.mean(rna) ) / np.std(rna)                                     
              
            if gene_data_norm=='NONE':
              genes_new [global_genes_processed,:] =  np.transpose(rna[1,:])    
            elif gene_data_norm=='GAUSSIAN':
              genes_new [global_genes_processed,:] =  np.transpose(normalized_rna[1,:])             
              if DEBUG>9:
                print( f"GENERATE:       INFO:          normalized_rna = \n\033[31m{normalized_rna}\033[m" )                      
            else:
              print( f"\033[31mGENERATE:      : FATAL:        no such gene data normalization mode as: {gene_data_norm} ... halting now[121]\033[m" ) 
              sys.exit(0)                 
                
            try:
              label = np.load(label_file)
              if DEBUG>99:
                print ( "GENERATE:      : label.shape =  \"{:}\"".format(  label.shape) )
                print ( "GENERATE:      : label       =  \"{:}\"".format(  label      ) )
              if DEBUG>999:
                print ( f"{label[0]},", end='', flush=True )
            except Exception as e:
              print ( "GENERATE:      :        ERROR: when opening this label file -- skipping\"{:}\"".format(e) )
              
            labels_new[global_genes_processed] =  label[0]
            if DEBUG>99:
              print ( f"{label[0]},", flush=True, end="" )
    
            gnames_new [global_genes_processed]  =  443                                                                           # Any old number. We don't currently use these
    
            if DEBUG>10:
              print ( "\nGENERATE:       INFO:         labels_new[{:}]".format( global_genes_processed ) )
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format( labels_new[global_genes_processed].size * labels_new[global_genes_processed].itemsize ) ) 
              print ( "GENERATE:       INFO:         value = {:}".format( labels_new[global_genes_processed] ) )
            if DEBUG>0:
              print ( "GENERATE:       INFO:         genes_new[{:}].shape = {:}".format( global_genes_processed,  genes_new[global_genes_processed].shape))
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format(genes_new[global_genes_processed].size * genes_new[global_genes_processed].itemsize))    
            if DEBUG>99:                                        
              print ( "GENERATE:       INFO:         value = \n{:}".format(genes_new[global_genes_processed] ) )  
            if DEBUG>999:
              print ( "GENERATE:       INFO:         gnames_new[{:}]".format( global_genes_processed ) )
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format( gnames_new[global_genes_processed].size * gnames_new[global_genes_processed].itemsize))
              print ( "GENERATE:       INFO:         value = {:}".format( gnames_new[global_genes_processed] ) )
             
            global_genes_processed+=1

      else:
        print( f"\033[31mGENERATE:      : FATAL:        no such mode: {input_mode} ... halting now[121]\033[m" ) 
        sys.exit(0)
    
    
    
  if not samples_processed-1==n_samples:
    print ( f"\033[31mGENERATE:      : FATAL:          total number of samples processed is not the same as the number required by the configuration variable 'n_samples'\033[m" )
    print ( f"GENERATE:      : FATAL:           total number of samples processed = {samples_processed-1}" )
    print ( f"GENERATE:      : FATAL:          'n_samples' (from variables.sh)   = {n_samples}" )
    print ( f"\033[31mGENERATE:      : FATAL:          halting now[134]\033[m" )
    sys.exit(0)
      
  print ( "\nGENERATE:       INFO:      finished processing:")       
  print ( "GENERATE:       INFO:      total number of samples  processed = \033[31m{:}\033[m".format(samples_processed-1))

  if input_mode=='image':
    print ( "GENERATE:       INFO:      user defined max tiles per image   = \033[31m{:}\033[m".format(n_tiles))
    print ( "GENERATE:       INFO:      total number of tiles processed    = \033[31m{:}\033[m".format(global_tiles_processed))     
    print ( "GENERATE:       INFO:      (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new )))
    print ( "GENERATE:       INFO:      (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 

  if input_mode=='rna':   
    print ( "GENERATE:       INFO:      (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( genes_new  )))
    print ( "GENERATE:       INFO:      (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new )))   

  print ( "GENERATE:       INFO:      (Numpy version of) labels_new (dummy data) ---------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( labels_new ))) 

  if DEBUG>999:  
      print ( f"GENERATE:       INFO:       (Numpy version of) labels_new = \n" )
      print ( f"{labels_new}", end='', flush=True ) 
    
  # convert everything into Torch style tensors

  if input_mode=='image':
    images_new   = torch.Tensor( images_new  )
    fnames_new   = torch.Tensor( fnames_new  )
    print( "GENERATE:       INFO:      finished converting image data from numpy array to Torch tensor") 

  if input_mode=='rna':   
    genes_new    = torch.Tensor( genes_new   )
    gnames_new   = torch.Tensor( gnames_new  )     
    print( "GENERATE:       INFO:      finished converting rna   data from numpy array to Torch tensor")

  labels_new  = torch.Tensor( labels_new).long()                                                         # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  print( "GENERATE:       INFO:      finished converting labels from numpy array to Torch tensor")
  labels_new.requires_grad_( False )                                                                      # labels aren't allowed gradients


  if input_mode=='image':
    print ( "GENERATE:       INFO:      shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "GENERATE:       INFO:      shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))

  if input_mode=='rna':   
    print ( "GENERATE:       INFO:      shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    print ( "GENERATE:       INFO:      shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))

  print ( "GENERATE:       INFO:      shape of (Torch version of) labels_new.size = {:}".format(labels_new.size() ))


  if DEBUG>99:     
    if input_mode=='image':   
      print ( {
          'images':  images_new,
          'fnames':  fnames_new,
          'tissues': labels_new,     
      } )
    elif input_mode=='rna':   
      print ( {
          'genes':   genes_new,
          'tissues': labels_new,
          'gnames':  gnames_new   
      } )
    else:
      pass      
  
  print( "GENERATE:       INFO:      now saving to Torch dictionary (this takes a little time)")

  if input_mode=='image':   
    torch.save({
        'images':  images_new,
        'fnames':  fnames_new,
        'tissues': labels_new,
    }, '%s/train.pth' % cfg.ROOT_DIR)
    
  elif input_mode=='rna':  
    torch.save({
          'genes':   genes_new,
          'tissues': labels_new,
          'gnames':  gnames_new  
    }, '%s/train.pth' % cfg.ROOT_DIR)
  else:
    pass


  print( "GENERATE:       INFO:      finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))
