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
import fnmatch
import random
import numpy as np
import pandas as pd

from  data.dlbcl_image.config import GTExV6Config

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
BLUE='\033[38;2;0;0;255m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;0;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1


def generate( args, n_samples, n_tiles, tile_size, n_genes, gene_data_norm, gene_data_transform ):

  # DON'T USE args.n_samples or args.n_tiles or args.gene_data_norm or args.tile_size since they are the job-level lists. Here we are just using one of each, passed in as the parameters above
  data_dir                = args.data_dir
  input_mode              = args.input_mode
  nn_mode                 = args.nn_mode
  rna_file_name           = args.rna_file_name
  rna_file_suffix         = args.rna_file_suffix
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  class_numpy_file_name   = args.class_numpy_file_name

  if input_mode=='image':
    print( f"{ORANGE}GENERATE:       INFO:      generate_image:(): input_mode is '{RESET}{CYAN}{input_mode}{RESET}{ORANGE}', so RNA data will not be generated{RESET}" )  


  print( "GENERATE:       INFO:      generate_image(): \
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
  slide_extension       = "svs"

  if DEBUG>0:
    print ( f"GENERATE:       INFO:        n_samples   = {n_samples}" )
    if input_mode=='image':  
      print ( f"GENERATE:       INFO:        n_tiles     = {n_tiles}" )      
      print ( f"GENERATE:       INFO:        total_tiles = {total_tiles}" )  
    if input_mode=='image':  
      print ( f"GENERATE:       INFO:        n_genes     = {n_genes}" )      

  cfg = GTExV6Config( 0,0 )

  if ( ( input_mode=='image' ) |  (nn_mode=='pre_compress' ) ):
    images_new   = np.empty( ( total_tiles,  3, tile_size, tile_size ), dtype=np.uint8   )                 #
    fnames_new   = np.empty( ( total_tiles                           ), dtype=np.int64    )                # np.int64 is equiv of torch.long
    labels_new   = np.empty( ( total_tiles,                          ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
    tiles_processed        =  0     # tiles processed per SVS image (directory)
    global_tiles_processed =  0     # global count of tiles processed 
  if ( ( input_mode=='rna' ) |  (nn_mode=='pre_compress' ) ):
    genes_new    = np.empty( ( n_samples, 1, n_genes                 ), dtype=np.float64 )                 #
    gnames_new   = np.empty( ( n_samples                             ), dtype=np.uint8   )                 # was gene names       NOT USED
    labels_new   = np.empty( ( n_samples,                            ), dtype=np.int_    )                 # labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 
    global_genes_processed =  0                                                                            # global count of genes processed
  
  samples_processed      = -1     # gobal count of samples processed (directories stepped into). Starting count is -1 because the top-level directory, which contains no images, is also traversed


  for dir_path, dirs, file_names in os.walk( data_dir ):                                                   # each iteration takes us to a new directory under data_dir

    tiles_processed         = 0
    samples_processed      += 1

    if samples_processed>n_samples:
      break

    if DEBUG>2:  
      print( "GENERATE:       INFO:      now processing directory \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-',   samples_processed, os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name

    # find the SVS file in each directory then  make and store an integer reference to it so for later retrieval when we are displaying tiles that belong to it in Tensorboard

    for f in sorted(file_names):                                                                           # examine every file in the current directory

       if f.endswith( slide_extension ):                                                                   # then we have the svs file for this directory (should only be one)
          svs_file_link_id   = abs(int(hash(f)//1000000000000))                                            # generate random string to use for the tensor link to the svs file name (can't have strings in tensors)
          svs_file_link_name = f"{svs_file_link_id:d}"

          fqsn = f"{dir_path}/entire_patch.npy"
          fqln = f"{data_dir}/{svs_file_link_name}.fqln"                                                   # name for the link
          try:
            os.symlink( fqsn, fqln)                                                                        # make the link
          except Exception as e:
            print ( f"{ORANGE}GENERATE:       NOTE:  Link already exists{RESET}" )
                
            
          if DEBUG>2:
              print( f"GENERATE:       INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}" )
              print( f"GENERATE:       INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'" )
              print (f"GENERATE:       INFO:  fully qualified file name of slide = '{MAGENTA}{fqsn}{RESET}'" )
              print (f"GENERATE:       INFO:                            data_dir = '{MAGENTA}{data_dir}{RESET}'" )              
              print (f"GENERATE:       INFO:    symlink for referencing the FQSN = '{MAGENTA}{fqln}{RESET}'" )


  if ( ( input_mode=='image' ) | ( nn_mode=='pre_compress' ) ):

    if DEBUG>1:
      if nn_mode=='pre_compress':
        print( f"{ORANGE}GENERATE:       INFO:      (image) nn_mode={MAGENTA}'pre_compress'{RESET}" )

    for dir_path, dirs, file_names in os.walk( data_dir ):                                                   # each iteration takes us to a new directory under data_dir
            
      for f in sorted(file_names):                                                                           # examine every file in the current directory
  
        if DEBUG>999:  
          print( f"GENERATE:       INFO:                    file_names = \n\033[31m{file_names}\033[m" )
  
        if DEBUG>4:
          if nn_mode=='pre_compress':
            print( f"{ORANGE}GENERATE:       INFO:      nn_mode={MAGENTA}'pre_compress'{RESET}" )  
            print( f"{ORANGE}GENERATE:       INFO:      tiles_processed={MAGENTA}{tiles_processed}{RESET}" )
            print( f"{ORANGE}GENERATE:       INFO:      n_tiles        ={MAGENTA}{n_tiles}{RESET}" )
  
          if ( tiles_processed<n_tiles ):                                                                    # while we have less than the requested number of tiles for this SVS image (directory)
            
            image_file    = os.path.join(dir_path, f)
            label_file    = os.path.join(dir_path, class_numpy_file_name)
            
            if DEBUG>0:
              if ( tiles_processed%10==0 ):
                print ("GENERATE:       INFO:          dir_path   = {:}".format(dir_path))
            
            if ( f.endswith('.' + tile_extension) & (not ( 'mask' in f ) ) & (not ( 'ized' in f ) )   ):     # because there may be other png files in each image folder besides the tile image files
    
              if DEBUG>0:
                if (    tiles_processed%(   int(  ( (n_tiles/10)//1 )  )   )    )==0:
                  print("GENERATE:       INFO:          about to process files {0:4d} to {1:4d} : for this image. Current file ({2:4d})  = \033[33m{3:s}\033[m".format( tiles_processed+1, tiles_processed+50, tiles_processed, image_file))
    
              try:
                print(f"GENERATE:       INFO:          attempting to read file {MAGENTA}{image_file}{RESET}")
                img = cv2.imread(image_file)
              except Exception as e:
                print ( "GENERATE:      :        ERROR: when opening this image file -- skipping \"{:}\"".format(e) )

              if DEBUG>0:
                if nn_mode=='pre_compress':
                  print( f"{GREEN}GENERATE:       INFO:      tiles_processed={MAGENTA}{global_tiles_processed}{RESET}" )
  
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

              if DEBUG>0:
                if nn_mode=='pre_compress':
                  print( f"{YELLOW}GENERATE:       INFO:      tiles_processed={MAGENTA}{global_tiles_processed}{RESET}" )

              labels_new[global_tiles_processed] =  label[0]                                                 # add it to the labels array
  
              if DEBUG>99:
                print ( f"{label[0]},", flush=True, end="" )
  
              fnames_new [global_tiles_processed]  =  svs_file_link_id                                      # link to filename of the slide from which this tile was extracted - see above
  
              if DEBUG>99:
                  print( f"GENERATE:       INFO: symlink for tile (fnames_new [{BLUE}{global_tiles_processed:3d}{RESET}]) = {BLUE}{fnames_new [global_tiles_processed]}{RESET}" )
              
    
              if DEBUG>9:
                print ( "=" *180)
                print ( "GENERATE:       INFO:          tile {:} for this image:".format( tiles_processed+1))
                print ( "GENERATE:       INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
                print ( "GENERATE:       INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
              if DEBUG>99:
                print ( "GENERATE:       INFO:                value = \n{:}".format(images_new[global_tiles_processed]))
      
              the_class=labels_new[global_tiles_processed]
              if the_class>3000:
                  print ( f"\033[31;1mGENERATE:       FATAL: Ludicrously large class value detected (class={the_class}) for tile '{image_file}'      HALTING NOW [1718]\033[m" )
                  sys.exit(0)
                  
              if DEBUG>9:
                size_in_bytes=labels_new[global_tiles_processed].size * labels_new[global_tiles_processed].itemsize
                print ( f"GENERATE:       INFO:            for labels_new[{global_tiles_processed}]; class={the_class}" )
      
              if DEBUG>99:
                print ( "GENERATE:       INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
                print ( "GENERATE:       INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
                print ( "GENERATE:       INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
               
              tiles_processed+=1
              global_tiles_processed+=1
              
            else:
              if DEBUG>1:
                print( "GENERATE:       INFO:          other file = \033[31m{:}\033[m".format( image_file ) ) 
  
          
  if ( ( input_mode=='rna' ) | ( nn_mode=='pre_compress' ) ):

    if DEBUG>1:
      print ( f"{ORANGE}GENERATE:       INFO:          (rna) input_mode = {MAGENTA}{input_mode}{RESET}", flush=True )
          
    samples_processed      = 0

    if DEBUG>1:
      print ( f"{ORANGE}GENERATE:       INFO:          (rna) data_dir = {MAGENTA}{data_dir}{RESET}", flush=True )
    
    for dir_path, dirs, file_names in os.walk( data_dir ):                                                   # each iteration takes us to a new directory under data_dir
  

      for f in sorted( file_names ):                                                                         # examine every file in the current directory

        if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
        
          if DEBUG>9:
            print ( f"GENERATE:       INFO:         f                        =  '{MAGENTA}{f}{RESET}' ",                        flush=True    )
            print ( f"GENERATE:       INFO:         rna_file_suffix          =  '{MAGENTA}{rna_file_suffix}{RESET}' ",          flush=True)
            print ( f"GENERATE:       INFO:         f.find(rna_file_suffix)  =  '{MAGENTA}{f.find(rna_file_suffix) }{RESET}' ", flush=True)
   
          if fnmatch.fnmatch( f, f"{rna_file_suffix}"   ):                                                                    # make sure it contains an rna file, because not all directories do. Some will only contain image files
        
            if DEBUG>0:
              print ( f"{PALE_ORANGE}GENERATE:       INFO:           file ending in '{MAGENTA}{rna_file_suffix}{RESET}{PALE_ORANGE}' was found{RESET}",                        flush=True    )
                                  
            rna_file      = os.path.join(dir_path, rna_file_name)                                             # it's in fact the numpy version of the rna file we're looking for
            label_file    = os.path.join(dir_path, class_numpy_file_name)
            
            try:
              rna = np.load( rna_file )
              if DEBUG>99:
                print ( f"GENERATE:       INFO:         rna.shape       =  '{CYAN}{rna.shape}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new.shape =  '{CYAN}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"GENERATE:       INFO:         rna             =  '{rna}' "            )
                print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
            except Exception as e:
              print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [118]{RESET}" )
              sys.exit(0)
                                                                          # remove row zero, which just holds the size of the file
            if DEBUG>999:  
              print( f"GENERATE:       INFO:                     rna = {CYAN}{rna}{RESET}" )              
            
            
            rna[np.abs(rna) < 1] = 0                                                                         # set all the values lower than 1 to be 0
            
            if gene_data_transform=='NONE':
              transformed_rna = rna                                  
            elif gene_data_transform=='LN':
              transformed_rna = np.log(rna) 
            elif gene_data_transform=='LOG2':
              transformed_rna = np.log2(rna) 
            elif gene_data_transform=='LOG2PLUS1':
              transformed_rna = np.log2(rna+1)
            elif gene_data_transform=='LOG10':
              transformed_rna = np.log10(rna)
            elif gene_data_transform=='LOG10PLUS1':
              transformed_rna = np.log10(rna+1)
            else:
              print( f"{RED}GENERATE:      : FATAL:        no such gene data transformation as: {gene_data_transform} ... halting now[184]{RESET}" )
              sys.exit(0) 

            if gene_data_norm=='NONE':
              normalized_rna =  transformed_rna 
            elif gene_data_norm=='GAUSSIAN':
              normalized_rna = ( transformed_rna - np.mean(transformed_rna) ) / np.std(transformed_rna)                                             
            else:
              print( f"{RED}GENERATE:      : FATAL:        no such gene normalization mode as: {gene_data_norm} ... halting now[184]{RESET}" )  
              sys.exit(0)       

            if DEBUG>0:
              print ( f"{DIM_YELLOW}GENERATE:       INFO: global_genes_processed = {global_genes_processed}{RESET}",  flush=True )
              
            genes_new [global_genes_processed] =  np.transpose(normalized_rna)               
              
            if DEBUG>99:
              print ( f"GENERATE:       INFO:         rna.shape       =  '{CYAN}{rna.shape}{RESET}' "      )
              print ( f"GENERATE:       INFO:         genes_new.shape =  '{CYAN}{genes_new.shape}{RESET}' ")
            if DEBUG>999:
              print ( f"GENERATE:       INFO:         rna             =  \n'{CYAN}{np.transpose(rna[1,:])}{RESET}' "      )
              print ( f"GENERATE:       INFO:         genes_new [{global_genes_processed}] =  '{CYAN}{genes_new[global_genes_processed]}{RESET}' ")                       
                
            try:
              label = np.load(label_file)
              if DEBUG>99:
                print ( "GENERATE:       INFO:         label.shape =  \"{:}\"".format(  label.shape) )
                print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
              if DEBUG>999:
                print ( f"{label[0]},", end='', flush=True )
            except Exception as e:
              print ( "GENERATE:      :        ERROR: when opening this label file -- skipping\"{:}\"".format(e) )
              
            labels_new[global_genes_processed] =  label[0]
            
            if DEBUG>99:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        labels_new[{CYAN}{global_genes_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
    
            gnames_new [global_genes_processed]  =  443                                                                           # Any old number. We don't currently use these
         
            if DEBUG>9:
              print ( "GENERATE:       INFO:         genes_new[{:}].shape = {:}".format( global_genes_processed,  genes_new[global_genes_processed].shape))
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format(genes_new[global_genes_processed].size * genes_new[global_genes_processed].itemsize))    
            if DEBUG>10:
              print ( "\nGENERATE:       INFO:         labels_new[{:}]".format( global_genes_processed ) )
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format( labels_new[global_genes_processed].size * labels_new[global_genes_processed].itemsize ) ) 
              print ( "GENERATE:       INFO:         value = {:}".format( labels_new[global_genes_processed] ) )
            if DEBUG>5:                                        
              print ( "GENERATE:       INFO:         value = \n{:}".format(genes_new[global_genes_processed] ) )  
            if DEBUG>999:
              print ( "GENERATE:       INFO:         gnames_new[{:}]".format( global_genes_processed ) )
              print ( "GENERATE:       INFO:         size in  bytes = {:,}".format( gnames_new[global_genes_processed].size * gnames_new[global_genes_processed].itemsize))
              print ( "GENERATE:       INFO:         value = {:}".format( gnames_new[global_genes_processed] ) )
             
            global_genes_processed += 1
            samples_processed      += 1

            if DEBUG>0:
              print ( f"{DIM_WHITE}GENERATE:       INFO: global_genes_processed = {CYAN}{global_genes_processed}{RESET}",  flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO: samples_processed      = {CYAN}{samples_processed}{RESET}",  flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO: n_samples              = {CYAN}{n_samples}{RESET}",               flush=True )
        
        if global_genes_processed>=n_samples:
          break 

  if ( ( input_mode=='rna' ) | ( nn_mode=='pre_compress' ) ):
    if not samples_processed==n_samples:
      print ( f"\033[31mGENERATE:      : WARNING:          total number of samples processed ({samples_processed}) does not equal configuration variable 'n_samples' ({n_samples})\033[m" )

  if input_mode=='image':
    if samples_processed-1<n_samples:
      print ( f"\033[31mGENERATE:      : FATAL:          total number of samples processed ({samples_processed-1}) is less than configuration variable 'n_samples' ({n_samples})halting now[134]\033[m" )
      sys.exit(0)
    if not samples_processed-1==n_samples:
      print ( f"\033[31mGENERATE:      : WARNING:          total number of samples processed ({samples_processed-1}) does not equal configuration variable 'n_samples' ({n_samples})\033[m" )

      
  print ( "GENERATE:       INFO:      finished processing:")
  if ( ( input_mode=='rna' ) | ( nn_mode=='pre_compress' ) ):
    print ( "GENERATE:       INFO:        total number of samples processed  = \033[31m{:}\033[m".format(samples_processed))
  else:
    print ( "GENERATE:       INFO:        total number of samples processed  = \033[31m{:}\033[m".format(samples_processed-1))

  if input_mode=='image':
    print ( "GENERATE:       INFO:        user defined tiles per sample      = \033[31m{:}\033[m".format(n_tiles))
    print ( "GENERATE:       INFO:        total number of tiles processed    = \033[31m{:}\033[m".format(global_tiles_processed))     
    print ( "GENERATE:       INFO:        (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new )))
    print ( "GENERATE:       INFO:        (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new ))) 

  if input_mode=='rna': 
    if args.nn_mode=='pre_compress':
      print ( "GENERATE:       INFO:        (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( genes_new  )))
      print ( "GENERATE:       INFO:        (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new )))   
    else:
      print ( "GENERATE:       INFO:        (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new )))
      print ( "GENERATE:       INFO:        (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new )))   


  print ( "GENERATE:       INFO:        (Numpy version of) labels_new (dummy data) ---------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( labels_new ))) 

  if DEBUG>0:  
      print ( f"GENERATE:       INFO:       (Numpy version of) size of labels_new = {CYAN}{labels_new.shape}{RESET}", flush=True )
  if DEBUG>99:  
      print ( f"GENERATE:       INFO:       (Numpy version of)         labels_new = \n" )
      print ( f"{CYAN}{labels_new}{RESET}", end='', flush=True )
    
  # convert everything into Torch style tensors

  if input_mode=='image':
    images_new   = torch.Tensor(images_new)
    fnames_new   = torch.Tensor(fnames_new).long()
    fnames_new.requires_grad_( False )
    print( "GENERATE:       INFO:        finished converting image data from numpy array to Torch tensor") 

  if input_mode=='rna':
    if args.nn_mode=='pre_compress':
      images_new   = torch.Tensor(images_new[0:n_samples])
      genes_new    = torch.Tensor( genes_new   )  
    else:
      genes_new    = torch.Tensor( genes_new   )
      gnames_new   = torch.Tensor( gnames_new  ) 
      gnames_new.requires_grad_( False )        
      print( "GENERATE:       INFO:        finished converting rna   data from numpy array to Torch tensor")

  labels_new  = torch.Tensor(labels_new[0:n_samples]).long()                                                         # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
  print( "GENERATE:       INFO:        finished converting labels from numpy array to Torch tensor")
  labels_new.requires_grad_( False )                                                                      # labels aren't allowed gradients


  if input_mode=='image':
    print ( "GENERATE:       INFO:        shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
    print ( "GENERATE:       INFO:        shape of (Torch version of) fnames_new.size  = {:}".format(fnames_new.size()   ))

  if input_mode=='rna':  
    if args.nn_mode=='pre_compress':
      print ( "GENERATE:       INFO:        shape of (Torch version of) images_new.size  = {:}".format(images_new.size()   ))
      print ( "GENERATE:       INFO:        shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
    else:
      print ( "GENERATE:       INFO:        shape of (Torch version of) genes_new.size   = {:}".format(genes_new.size()     ))
      print ( "GENERATE:       INFO:        shape of (Torch version of) gnames_new.size  = {:}".format(gnames_new.size()   ))

  print ( "GENERATE:       INFO:        shape of (Torch version of) labels_new.size = {:}".format(labels_new.size() ))

  if DEBUG>999:     
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
  
  print( "GENERATE:       INFO:        now saving to Torch dictionary (this takes a little time)")

  if input_mode=='image': 
    torch.save({
        'images':  images_new,
        'fnames':  fnames_new,
        'tissues': labels_new,
    }, '%s/train.pth' % cfg.ROOT_DIR)
    
  elif input_mode=='rna':  
    if args.nn_mode=='pre_compress':
      torch.save({
            'images':  images_new,
            'genes':   genes_new,
            'tissues': labels_new,
      }, '%s/train.pth' % cfg.ROOT_DIR)
    else:
      torch.save({
            'genes':   genes_new,
            'tissues': labels_new,
            'gnames':  gnames_new  
      }, '%s/train.pth' % cfg.ROOT_DIR)
  else:
    pass

  print( "GENERATE:       INFO:      finished saving Torch dictionary to \033[31m{:}/train.pth\033[m".format(cfg.ROOT_DIR))
