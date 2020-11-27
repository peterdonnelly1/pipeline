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
MIKADO='\033[38;2;255;196;12m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

SUCCESS=1
DEBUG=1


def generate( args, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform ):

  # DON'T USE args.n_samples or args.n_tiles or args.gene_data_norm or args.tile_size since these are job-level lists. Here we are just using one value of each, passed in as the parameters above
  data_dir                = args.data_dir
  input_mode              = args.input_mode
  rna_file_name           = args.rna_file_name
  rna_file_suffix         = args.rna_file_suffix  
  rna_file_reduced_suffix = args.rna_file_reduced_suffix
  class_numpy_file_name   = args.class_numpy_file_name
  use_autoencoder_output  = args.use_autoencoder_output
  use_unfiltered_data     = args.use_unfiltered_data  

  if DEBUG>0:
    print( "GENERATE:       INFO:   \
   input_mode=\033[36;1m{:}\033[m,\
   data_dir=\033[36;1m{:}\033[m,\
   n_samples=\033[36;1m{:}\033[m,\
   n_tiles=\033[36;1m{:}\033[m,\
   tile_size=\033[36;1m{:}\033[m,\
   gene_data_norm=\033[36;1m{:}\033[m,\
   gene_data_transform=\033[36;1m{:}\033[m,\
   rna_file_name=\033[36;1m{:}\033[m,\
   class_numpy_file_name=\033[36;1m{:}\033[m,\
   n_tiles=\033[36;1m{:}\033[m"\
  .format( input_mode, data_dir, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform, rna_file_name, class_numpy_file_name, n_tiles), flush=True )
 
  tiles_required = n_samples*n_tiles

  cfg = GTExV6Config( 0,0 )

  #print ( f"\033[36B",  flush=True ) 


  # (1) analyse dataset directory

  if use_unfiltered_data=='True':
    rna_suffix = rna_file_suffix[1:]
  else:
    rna_suffix = rna_file_reduced_suffix

  cumulative_image_file_count = 0
  cumulative_png_file_count   = 0
  cumulative_rna_file_count   = 0
  cumulative_other_file_count = 0
  
  for dir_path, dirs, files in os.walk( data_dir ):                                                        # each iteration takes us to a new directory under data_dir

    if not (dir_path==data_dir):                                                                           # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data      
      
      image_file_count   = 0
      rna_file_count     = 0
      png_file_count     = 0
      other_file_count   = 0

      for f in sorted( files ):
       
        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
          image_file_count            +=1
          cumulative_image_file_count +=1
        elif  ( f.endswith( 'png' ) ):
          png_file_count              +=1
          cumulative_png_file_count   +=1
        elif  ( f.endswith( rna_suffix ) ):
          rna_file_count              +=1
          cumulative_rna_file_count   +=1
        else:
          other_file_count            +=1
          cumulative_other_file_count +=1
        
      if DEBUG>77:
        if ( ( rna_file_count>1 ) | ( image_file_count>1 ) ): 
          print( f"GENERATE:       INFO:    \033[58Cdirectory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq file(s) and {MIKADO}{image_file_count:<2d}{RESET} image files(s) and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )
          time.sleep(0.5)       
        else:
          print( f"GENERATE:       INFO:    directory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq files, {MIKADO}{image_file_count:<2d}{RESET} image files and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )

  if DEBUG>9:
    print( f"GENERATE:       INFO:      image_file_count          = {MIKADO}{cumulative_image_file_count:<6d}{RESET}", flush=True  )
    print( f"GENERATE:       INFO:      cumulative_png_file_count = {MIKADO}{cumulative_png_file_count:<6d}{RESET}", flush=True  )
    print( f"GENERATE:       INFO:      rna_file_count            = {MIKADO}{cumulative_rna_file_count:<6d}{RESET}",   flush=True  )
    print( f"GENERATE:       INFO:      other_file_count          = {MIKADO}{cumulative_other_file_count:<6d}{RESET}", flush=True  )






  # (2) process IMAGE data if applicable
  
  #  (2A) set up numpy data structures to accumulate image data
  
  if ( input_mode=='image' ) :
    images_new      = np.ones ( ( tiles_required,  3, tile_size, tile_size ), dtype=np.uint8   )              
    fnames_new      = np.zeros( ( tiles_required                           ), dtype=np.int64   )              # np.int64 is equiv of torch.long
    img_labels_new  = np.zeros( ( tiles_required,                          ), dtype=np.int_    )              # img_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training 


  #  (2B) Process image data. The work is done in process_image_files()

  if ( input_mode=='image' ):

    # check to see that there actually are tiles to process    
    if cumulative_png_file_count==0:
      print ( f"{RED}GENERATE:       FATAL:  there are no tile files ('png' files) at all. To generate tiles, run '{CYAN}./do_all.sh <cancer type code> image{RESET}{RED}' ... halting now{RESET}", flush=True )                 
      sys.exit(0)         

    if ( input_mode=='image' ):
      print( f"{ORANGE}GENERATE:       NOTE:  input_mode is '{RESET}{CYAN}{input_mode}{RESET}{ORANGE}', so rna and other data will not be generated{RESET}" )  
    
    # process image data
    tiles_processed       = 0
    directories_processed = 0


    #(2C) traverse dataset and process each directory
    
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration of os.walk takes us to a new directory under data_dir    

      if DEBUG>1:
        print ( f"{WHITE}GENERATE:       INFO: dir_path = {BITTER_SWEET}{dir_path}{RESET}",  flush=True )      

      if DEBUG>8:
        print( f"GENERATE:       INFO:     dir_path               = {MAGENTA}{dir_path}{RESET}",                  flush=True     )

      # does the work
      tiles_processed = process_image_files ( args, dir_path, dirs, files, images_new, img_labels_new, fnames_new, n_tiles, tiles_processed )

      directories_processed+=1
      if DEBUG>88:
        print( f"GENERATE:       INFO:     directories_processed  = {BLEU}{directories_processed-1:<8d}{RESET}",  flush=True       )
        print( f"GENERATE:       INFO:     tiles_processed        = {BLEU}{tiles_processed:<8d}{RESET}",        flush=True       ) 
        print( f"GENERATE:       INFO:     tiles required         = {BLEU}{tiles_required:<8d}{RESET}",         flush=True       )             
      if tiles_processed>=tiles_required:
        break




  # (3) process "IMAGE_RNA" data (concatenated embeddings) if applicable 

  if ( input_mode=='image_rna' ):

    print( f"GENERATE:       NOTE:  input_mode is '{RESET}{CYAN}{input_mode}{RESET}'" ) 
    
      
    # (3A) create concatenated embeddings

    dir_has_rna_data                           = False
    dirs_which_have_rna_data                   = 0
    dir_also_has_image                         = False
    dirs_which_have_matched_image_rna_files    = 0


    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>888:  
        print( f"{DIM_WHITE}GENERATE:       INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
                
        dir_has_rna_data    = False
        dir_also_has_image  = False
  
        for f in sorted( files ):
          if  ( f.endswith( args.embedding_file_suffix_rna  ) ):
            dir_has_rna_data=True
            rna_file  = f
          if ( ( f.endswith( 'svs' ))  |  ( f.endswith( 'tif' ) )  |  ( f.endswith( 'tiff' ) )  ):
            dir_also_has_image=True
                  
        if dir_has_rna_data & dir_also_has_image:
          if DEBUG>6:
            print ( f"{WHITE}GENERATE:       INFO:   case {CYAN}{data_dir}/{os.path.basename(dir_path)}{RESET} \r\033[100C has both matched and rna files (listed above) (count= {MIKADO}{dirs_which_have_matched_image_rna_files+1}{RESET})",  flush=True )
          dirs_which_have_matched_image_rna_files+=1
          for f in sorted( files ):
            if f.endswith( args.embedding_file_suffix_image ):
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image embedding file        =  {ARYLIDE}{f}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}GENERATE:       INFO:   rna   embedding file        =  {BLEU}{rna_file}{RESET}",  flush=True )
              image_embedding = np.load ( f"{dir_path}/{f}" )
              rna_embedding   = np.load ( f"{dir_path}/{rna_file}" )
              #rna_embedding = np.zeros_like( rna_embedding )                                              # force rna-seq portion of the concatenated embedding to zero for testing purposes
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image_embedding.shape       =  {ARYLIDE}{image_embedding.shape}{RESET}",  flush=True )
                print ( f"{DIM_WHITE}GENERATE:       INFO:   rna_embedding.shape         =  {BLEU}{rna_embedding.shape}{RESET}",  flush=True )
              
              image_rna_embedding = np.concatenate( (image_embedding, rna_embedding ), axis=0)
              if DEBUG>6:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   image_rna_embedding.shape   =  {ARYLIDE}{image_rna_embedding.shape}{RESET}",  flush=True )
              random_name   = f"_{random.randint(10000000, 99999999)}_image_rna_matched___image_rna"
              save_fqn      = f"{dir_path}/{random_name}"
              if DEBUG>8:
                print ( f"{DIM_WHITE}GENERATE:       INFO:   saving concatenated embedding {PINK}{save_fqn}{RESET}",  flush=True )
              if  ( args.just_test=='False' ):                                            # only create concatenated embeddings in training mode. Nonetheless in test mode we need the value of 'dirs_which_have_matched_image_rna_files'
                np.save ( save_fqn, image_rna_embedding )
              

    # (3B) determine 'n_genes' (sic) by looking at an (any) image_rna file, (so that it doesn't have to be manually entered as a user parameter)
    if use_autoencoder_output=='False':
   
      if DEBUG>0:
        print ( f"GENERATE:       INFO:  about to determine length of an image_rna embedding"      )
    
      found_one=False
      for dir_path, dirs, files in os.walk( data_dir ):                                                    # each iteration takes us to a new directory under data_dir
        if not (dir_path==data_dir):                                                                       # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
          
          if check_mapping_file( args, dir_path ) == True:
          
            for f in sorted(files):                                                                        # examine every file in the current directory
              if found_one==True:
                break
              if  ( f.endswith( args.embedding_file_suffix_image_rna  ) ):
                if DEBUG>999:
                  print (f)     
                image_rna_file      = os.path.join(dir_path, f )
                try:
                  image_rna = np.load( image_rna_file )
                  n_genes=image_rna.shape[0]
                  found_one=True
                  if DEBUG>0:
                    print ( f"GENERATE:       INFO:   image_rna.shape      =  '{MIKADO}{image_rna.shape}{RESET}' "      )
                  if DEBUG>0:
                    print ( f"GENERATE:       INFO:  n_genes (determined)  = {MIKADO}{n_genes}{RESET}"        )
                except Exception as e:
                    print ( f"{RED}GENERATE:             FATAL: '{e}'{RESET}" )
                    print ( f"{RED}GENERATE:                          Explanation: a required image_rna embedding file doesn't exist. (Probably no image_rna files exist){RESET}" )                 
                    print ( f"{RED}GENERATE:                          Did you change to image_rna mode from another input mode but neglect to run '{CYAN}./do_all.sh{RESET}{RED}' to generate the image_rna files the NN needs for image_rna mode ? {RESET}" )
                    print ( f"{RED}GENERATE:                          If so, run '{CYAN}./do_all.sh <cancer type code> image_rna{RESET}{RED}' to generate the image_rna files{RESET}" )                 
                    print ( f"{RED}GENERATE:                          Halting now{RESET}" )                 
                    sys.exit(0)

      if found_one==False:                  
        print ( f"{RED}GENERATE:       FATAL: No image_rna embedding files exist in the dataset directory ({MAGENTA}{data_dir}{RESET}{RED})"                                                                          )                 
        print ( f"{RED}GENERATE:                 Possible explanations:{RESET}"                                                                                                                       )
        print ( f"{RED}GENERATE:                   (1) Did you change to {CYAN}image_rna{RESET}{RED} mode from another input mode but neglect to regenerate the files input required for {CYAN}image_rna{RESET}{RED} mode ?" )
        print ( f"{RED}GENERATE:                 Possible remedies:{RESET}"                                                                                                                       )
        print ( f"{RED}GENERATE:                       (A) (easist, but regenerates everything) run '{CYAN}./do_all_image_rna.sh.sh  -d <cancer_type_code>{RESET}{RED}" )
        print ( f"{RED}GENERATE:                       (B) (faster) run '{CYAN}./do_all.sh     -d <cancer_type_code> -i rna{RESET}{RED}' to train the rna model'" )                 
        print ( f"{RED}GENERATE:                               then run '{CYAN}./just_test.sh  -d <cancer_type_code> -i rna  -m image_rna <cancer_type_code> image_rna{RESET}{RED}' to generate the rna embedding files'" )                 
        print ( f"{RED}GENERATE:                               then run '{CYAN}./do_all.sh     -d <cancer_type_code> -i image_rna{RESET}{RED}' to generate the image_rna embedding files. " )                 
        print ( f"{RED}GENERATE:               Halting now{RESET}" )                 
        sys.exit(0)
    
                


    # (3C) set up numpy data structures to accumulate image_rna data as it is processed 

    # need to know total_number_of_image_rna_files in advance to be able to create numpy array to hold them. Determine using this rule: one concatenated image_rna file (___image_rna.npy) will be created for every existing IMAGE embedding file in a directory that has both an image embedding file (___image.npy)and an rna embedding file (___rna.npy)         
    if DEBUG>0:
      print ( f"{ORANGE}GENERATE:       INFO:   dirs_which_have_matched_image_rna_files         =  {ARYLIDE}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )
      print ( f"{ORANGE}GENERATE:       INFO:   n_tiles                                         =  {ARYLIDE}{n_tiles}{RESET}",                                  flush=True )
    total_number_of_image_rna_files = dirs_which_have_matched_image_rna_files * n_tiles
    if DEBUG>0:
      print ( f"{ORANGE}GENERATE:       INFO:   (hence) total_number_of_image_rna_files         =  {ARYLIDE}{total_number_of_image_rna_files}{RESET}",  flush=True )


    if ( input_mode=='image_rna' ):
      if use_autoencoder_output=='False':
        genes_new      = np.zeros( ( total_number_of_image_rna_files, 1, n_genes                ), dtype=np.float64 )
      fnames_new       = np.zeros( ( total_number_of_image_rna_files                            ), dtype=np.int64   )              
      gnames_new       = np.zeros( ( total_number_of_image_rna_files                            ), dtype=np.uint8   )            # was gene names                                               NOT USED
      global_image_rna_files_processed =  0                                                                # global count of genes processed
      rna_labels_new   = np.zeros( ( total_number_of_image_rna_files,                           ), dtype=np.int_    )            # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training


    # (3D) process image_rna data
    
    # process image_rna data
    symlinks_created               = 0
            
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under data_dir
 
      if DEBUG>9:  
        print( f"{DIM_WHITE}GENERATE:       INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
        
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not dat

        # ~ # (3Di) Make and store a  softlink based on an integer reference to the case id for later use so that DENSE will later know where to save the rna-seq embeddings (if this option is enabled)
        
        # ~ for f in sorted( files ):  
            
          # ~ if  f.endswith( args.embedding_file_suffix_image_rna ):
            
            # ~ image_rna_file_link_id   = random.randint(1000000, 9999999)                                    # generate random string to use for the softlink to the file name (can't have strings in tensors)
            # ~ image_rna_file_link_name = f"{image_rna_file_link_id:d}"
      
            # ~ fqcd = f"{dir_path}"                                                                           # fully qualified case directory
            # ~ fqln = f"{args.data_dir}/{image_rna_file_link_name}.fqln"                                      # fully qualified link name
            # ~ try:
              # ~ os.symlink( fqcd, fqln)                                                                      # make a link from fqln to fqcd
              # ~ symlinks_created +=1
              # ~ if DEBUG>88:
                # ~ print ( f"GENERATE:          ALL GOOD:          softlink (fqln) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )
            # ~ except Exception as e:
              # ~ print ( f"{RED}GENERATE:       EXCEPTION RAISED:  softlink (fqln) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )
              
            # ~ if DEBUG>0:
              # ~ print( f"GENERATE:       INFO:                      image_rna_file_link_id =  {MAGENTA}{image_rna_file_link_id}{RESET}" )
              # ~ print( f"GENERATE:       INFO:                    image_rna_file_link_name = '{MAGENTA}{image_rna_file_link_name}{RESET}'" )
              # ~ print (f"GENERATE:       INFO: fully qualified file name of case directory = '{MAGENTA}{fqcd}{RESET}'" )        
              # ~ print (f"GENERATE:       INFO:            symlink for referencing the fqcd = '{MAGENTA}{fqln}{RESET}'" )
              # ~ print (f"GENERATE:       INFO:                     symlinks created so far = '{MAGENTA}{symlinks_created}{RESET}'" )


        # (3Dii) Process the rna-seq file
        
        for f in sorted( files ):              

          if  f.endswith( args.embedding_file_suffix_image_rna  ):
            
            dir_has_image_rna_data = True
            image_rna_file         = os.path.join( dir_path, f         )
            label_file             = os.path.join( dir_path, class_numpy_file_name )

            # set up the pytorch array
            if DEBUG>8:
              print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {BLEU}{f}{RESET}", flush=True )
            
            if use_autoencoder_output=='False':                                                            # skip gene processing (but do labels and gnames) if we're using autoencoder output 
          
              try:
                image_rna_embedding = np.load( image_rna_file )
                if DEBUG>9:
                  print ( f"GENERATE:       INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                  print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"GENERATE:       INFO:         image_rna_embedding             =  '{image_rna_embedding}' "            )
                  print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [118]{RESET}" )
                sys.exit(0)
          
          
              if DEBUG>999:  
                print( f"GENERATE:       INFO:                     image_rna_embedding = {CYAN}{image_rna_embedding}{RESET}" )              

          
              genes_new [global_image_rna_files_processed] =  np.transpose(image_rna_embedding)
                
              if DEBUG>99:
                print ( f"GENERATE:       INFO:         image_rna_embedding.shape       =  '{MIKADO}{image_rna_embedding.shape}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"GENERATE:       INFO:         image_rna_embedding             =  \n'{MIKADO}{np.transpose(image_rna_embedding[1,:])}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new [{global_image_rna_files_processed}] =  '{CYAN}{genes_new[global_image_rna_files_processed]}{RESET}' ")                       
              
            try:
              label = np.load( label_file)
              if DEBUG>6:
                print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
              if DEBUG>2:
                print ( f"{label[0]},", end='', flush=True )
            except Exception as e:
              print ( f"{RED}TRAINLENEJ:     FATAL: '{e}'{RESET}" )
              print ( f"{RED}TRAINLENEJ:     FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{RED}{RESET}" )
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{RED}' to regenerate them{RESET}" ) 
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{RED}', because without it, no class files can be generated'{RESET}" )                                    
              print ( f"{RED}TRAINLENEJ:     FATAL:  cannot continue - halting now{RESET}" )                 
              sys.exit(0)     
              
            rna_labels_new[global_image_rna_files_processed] =  label[0]

            
            if DEBUG>6:
              print ( f"{DIM_WHITE}GENERATE:       INFO:         label[0][{MIKADO}{global_image_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
          
          
           # fnames_new [global_image_rna_files_processed  ]  =  image_rna_file_link_id                                                            # link to folder from which that this image_rna_embedding sample belongs to - passed in as a parameter
            fnames_new [global_image_rna_files_processed  ]  =  777   
          
            if DEBUG>888:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        image_rna_file_link_id = {MIKADO}{image_rna_file_link_id}{RESET}",                          flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO:        fnames_new[{MIKADO}{global_image_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_image_rna_files_processed  ]}{RESET}", flush=True )
            
            
            gnames_new [global_image_rna_files_processed]  =  781                                                                           # Any old number. We don't currently use these
          
            if DEBUG>888:
              print ( f"{WHITE}GENERATE:       INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
              time.sleep(.4)  

            global_image_rna_files_processed+=1

            if DEBUG>9:
              print ( f"{WHITE}GENERATE:       INFO: global_image_rna_files_processed = {MIKADO}{global_image_rna_files_processed}{RESET}",  flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO: n_samples                  = {CYAN}{n_samples}{RESET}",               flush=True )

          

          
              
  
              
              
  
          
  if ( input_mode=='rna' ):


    # (4A) determine 'n_genes' by looking at an (any) rna file, (so that it doesn't have to be manually entered as a user parameter)
    if use_autoencoder_output=='False':
  
      # To determine n_genes, (so that it doesn't have to be manually specified), need to examine just ONE of the rna files   
      if DEBUG>2:
        print ( f"GENERATE:       INFO:  about to determine value of 'n_genes'"      )
    
      found_one=False
      for dir_path, dirs, files in os.walk( data_dir ):                                                    # each iteration takes us to a new directory under data_dir
        if not (dir_path==data_dir):                                                                       # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
          
          if check_mapping_file( args, dir_path ) == True:
          
            for f in sorted(files):                                                                        # examine every file in the current directory
              if found_one==True:
                break
              if ( f.endswith( rna_file_suffix[1:]) ):                                                     # have to leave out the asterisk apparently
                if DEBUG>999:
                  print (f)     
                rna_file      = os.path.join(dir_path, rna_file_name)
                try:
                  rna = np.load( rna_file )
                  n_genes=rna.shape[0]
                  found_one=True
                  if DEBUG>9:
                    print ( f"GENERATE:       INFO:   rna.shape             =  '{MIKADO}{rna.shape}{RESET}' "      )
                  if DEBUG>2:
                    print ( f"GENERATE:       INFO:  n_genes (determined)  = {MIKADO}{n_genes}{RESET}"        )
                except Exception as e:
                    print ( f"{RED}GENERATE:             FATAL: '{e}'{RESET}" )
                    print ( f"{RED}GENERATE:                          Explanation: a required rna file doesn't exist. (Probably no rna files exist){RESET}" )                 
                    print ( f"{RED}GENERATE:                          Did you change from image mode to rna mode but neglect to run '{CYAN}./do_all.sh{RESET}{RED}' to generate the rna files the NN needs for rna mode ? {RESET}" )
                    print ( f"{RED}GENERATE:                          If so, run '{CYAN}./do_all.sh <cancer type code> rna{RESET}{RED}' to generate the rna files{RESET}" )                 
                    print ( f"{RED}GENERATE:                          Halting now{RESET}" )                 
                    sys.exit(0)

      if found_one==False:                  
        print ( f"{RED}GENERATE:       FATAL: No rna files at all exist in the dataset directory ({MAGENTA}{data_dir}{RESET}{RED})"                                                                          )                 
        print ( f"{RED}GENERATE:                 Possible explanations:{RESET}"                                                                                                                       )
        print ( f"{RED}GENERATE:                   (1) The dataset '{CYAN}{args.dataset}{RESET}{RED}' doesn't have any rna-seq data. It might only have image data{RESET}" )
        print ( f"{RED}GENERATE:                   (2) Did you change from image mode to rna mode but neglect to run '{CYAN}./do_all.sh{RESET}{RED}' to generate the files required for rna mode ? {RESET}" )
        print ( f"{RED}GENERATE:                       If so, run '{CYAN}./do_all.sh <cancer_type_code> rna{RESET}{RED}' to generate the rna files{RESET}{RED}. After that, you will be able to use '{CYAN}./just_run.sh <cancer_type_code> rna{RESET}{RED}'" )                 
        print ( f"{RED}GENERATE:               Halting now{RESET}" )                 
        sys.exit(0)


    # (4B) set up numpy data structures to accumulate rna data as it is processed 

    if ( input_mode=='rna' ):
      # set up numpy data structures to accumulate rna-seq data as it is processed    
      if use_autoencoder_output=='False':
        genes_new      = np.zeros( ( n_samples, 1, n_genes                ), dtype=np.float64 )
      fnames_new       = np.zeros( ( n_samples                            ), dtype=np.int64   )              
      gnames_new       = np.zeros( ( n_samples                            ), dtype=np.uint8   )              # was gene names                                               NOT USED
      global_rna_files_processed =  0                                                                        # global count of genes processed
      rna_labels_new   = np.zeros( ( n_samples,                           ), dtype=np.int_    )              # rna_labels_new holds class label (integer between 0 and Number of classes-1). Used as Truth labels by Torch in training


    # (4C) process rna-seq data
    
    # info and warnings
    if ( input_mode=='rna' ):
      print( f"GENERATE:       NOTE:  input_mode is '{RESET}{CYAN}{input_mode}{RESET}', so image and other data will not be generated{RESET}" )  

    if use_unfiltered_data=='True':
      rna_suffix = rna_file_suffix[1:]
      print( f"{ORANGE}GENERATE:       NOTE:  flag {CYAN}'USE_UNFILTERED_DATA'{CYAN}{RESET}{ORANGE} is set, so all genes listed in file '{MAGENTA}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}{ORANGE}' will be used{RESET}" )        
    else:
      rna_suffix = rna_file_reduced_suffix
      print( f"{ORANGE}GENERATE:       NOTE:  only the set of genes specified in the configuration setting '{CYAN}TARGET_GENES_REFERENCE_FILE{RESET}' will be used. Set {CYAN}'USE_UNFILTERED_DATA'{CYAN} to True if you wish to use all '{MAGENTA}ENSG_UCSC_biomart_ENS_id_to_gene_name_table{RESET}' genes" ) 
   

    # process rna-seq data
    dir_has_rna_data               = False
    dirs_which_have_rna_data       = 0
    dir_also_has_image             = False                                                                 # ditto
    dirs_which_have_matched_image_rna_files    = 0                                                                     # used for image_rna mode, where we only want to use cases (dirs) which contain both rna and image data
    symlinks_created               = 0
            
    for dir_path, dirs, files in os.walk( data_dir ):                                                      # each iteration takes us to a new directory under data_dir
 
      if DEBUG>2:  
        print( "GENERATE:       INFO:      now processing directory \033[31;1m{:} {:} {:}\033[m".format( ( len(dir_path.split(os.sep)) - 4) * '-', os.path.basename(dir_path)))               # one dash for the highest directory, a further dash for each subdirectory; then current directory name
  
      if not (dir_path==data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
                
        dir_has_rna_data    = False
        dir_also_has_image  = False

        for f in sorted( files ):                                                                          # see if the directory also has rna files (later for use in 'image_rna' mode)
         
          if  ( f.endswith( rna_suffix ) ):
            dir_has_rna_data=True
          if ( ( f.endswith( 'svs' ))  |  ( f.endswith( 'tif' ) )  |  ( f.endswith( 'tiff' ) )  ):
            dir_also_has_image=True
        
        for f in sorted(files):                                                                            # examine EVERY file in the current directory

          if DEBUG>8:
            print ( f"{DIM_WHITE}GENERATE:       INFO:  rna_suffix                   = {MIKADO}{rna_suffix}{RESET}", flush=True )
            print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {MAGENTA}{f}{RESET}", flush=True )            
                                  

        # (4Ci) Make and store a  softlink based on an integer reference to the case id for later use so that DENSE will later know where to save the rna-seq embeddings (if this option is enabled)
            
          if  f == rna_file_name:
            
            rna_file_link_id   = random.randint(1000000, 9999999)                                          # generate random string to use for the softlink to the file name (can't have strings in tensors)
            rna_file_link_name = f"{rna_file_link_id:d}"
      
            fqcd = f"{dir_path}"                                                                           # fully qualified case directory
            fqln = f"{args.data_dir}/{rna_file_link_name}.fqln"                                            # fully qualified link name
            try:
              os.symlink( fqcd, fqln)                                                                      # make a link from fqln to fqcd
              symlinks_created +=1
              if DEBUG>88:
                print ( f"GENERATE:       ALL GOOD:          softlink (FQLN) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )
            except Exception as e:
              print ( f"{RED}GENERATE:       EXCEPTION RAISED:  softlink (FQLN) = {MIKADO}{fqln}{RESET} \r\033[100C and target (fqcd) = {MIKADO}{fqcd}{RESET}" )

                
            if DEBUG>2:
              print( f"GENERATE:       INFO:                            rna_file_link_id =  {MAGENTA}{rna_file_link_id}{RESET}" )
              print( f"GENERATE:       INFO:                          rna_file_link_name = '{MAGENTA}{rna_file_link_name}{RESET}'" )
              print (f"GENERATE:       INFO: fully qualified file name of case directory = '{MAGENTA}{fqcd}{RESET}'" )        
              print (f"GENERATE:       INFO:            symlink for referencing the fqcd = '{MAGENTA}{fqln}{RESET}'" )
              print (f"GENERATE:       INFO:                     symlinks created so far = '{MAGENTA}{symlinks_created}{RESET}'" )


          # (4Cii) Process the rna-seq file
          
          if  f == rna_file_name:
            
            dir_has_rna_data = True
            rna_file      = os.path.join( dir_path, rna_file_name         )
            label_file    = os.path.join( dir_path, class_numpy_file_name )

            # set up the pytorch array
          
            if DEBUG>8:
              print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {BLEU}{f}{RESET}", flush=True )
            
            if use_autoencoder_output=='False':                                                                      # Skip gene processing (but do labels and gnames) if we're using autoencoder output. We'll LATER load and use ae output file as genes_new rather than process raw rna-seq data 
          
              try:
                rna = np.load( rna_file )
                if DEBUG>9:
                  print ( f"GENERATE:       INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                  print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
                if DEBUG>999:
                  print ( f"GENERATE:       INFO:         rna             =  '{rna}' "            )
                  print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
              except Exception as e:
                print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [118]{RESET}" )
                sys.exit(0)
          
          
              if DEBUG>999:  
                print( f"GENERATE:       INFO:                     rna = {CYAN}{rna}{RESET}" )              
              
              rna[np.abs(rna) < 1] = 0                                                                               # set all the values lower than 1 to be 0
              
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
                print( f"{RED}GENERATE:      FATAL:        no such gene data transformation as: {gene_data_transform} ... halting now[184]{RESET}" )
                sys.exit(0) 
          
              if gene_data_norm=='NONE':
                normalized_rna =  transformed_rna
              elif gene_data_norm=='JUST_SCALE':
                normalized_rna = transformed_rna / np.std(transformed_rna)   
              elif gene_data_norm=='GAUSSIAN':
                normalized_rna = ( transformed_rna - np.mean(transformed_rna) ) / np.std(transformed_rna)                                             
              else:
                print( f"{RED}GENERATE:      FATAL:        no such gene normalization mode as: {gene_data_norm} ... halting now[378]{RESET}" )  
                sys.exit(0)       
          
              genes_new [global_rna_files_processed] =  np.transpose(normalized_rna)               
                
              if DEBUG>99:
                print ( f"GENERATE:       INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
              if DEBUG>999:
                print ( f"GENERATE:       INFO:         rna             =  \n'{MIKADO}{np.transpose(rna[1,:])}{RESET}' "      )
                print ( f"GENERATE:       INFO:         genes_new [{global_rna_files_processed}] =  '{CYAN}{genes_new[global_rna_files_processed]}{RESET}' ")                       
              
            try:
              label = np.load( label_file)
              if DEBUG>99:
                print ( "GENERATE:       INFO:         label.shape =  \"{:}\"".format(  label.shape) )
                print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
              if DEBUG>2:
                print ( f"{label[0]},", end='', flush=True )
            except Exception as e:
              print ( f"{RED}TRAINLENEJ:     FATAL: '{e}'{RESET}" )
              print ( f"{RED}TRAINLENEJ:     FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{RED}{RESET}" )
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{RED}' to regenerate them{RESET}" ) 
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
              print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{RED}', because without it, no class files can be generated'{RESET}" )                                    
              print ( f"{RED}TRAINLENEJ:     FATAL:  cannot continue - halting now{RESET}" )                 
              sys.exit(0)     
              
            rna_labels_new[global_rna_files_processed] =  label[0]
            #rna_labels_new[global_rna_files_processed] =  random.randint(0,5)                        ################### swap truth labels to random numbers for testing purposes
            
            if DEBUG>777:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_labels_new[{MIKADO}{global_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )
          
          
            fnames_new [global_rna_files_processed  ]  =  rna_file_link_id                                                            # link to folder from which that this rna sample belongs to - passed in as a parameter
          
            if DEBUG>888:
              print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_file_link_id = {MIKADO}{rna_file_link_id}{RESET}",                          flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO:        fnames_new[{MIKADO}{global_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_rna_files_processed  ]}{RESET}", flush=True )
            
            
            gnames_new [global_rna_files_processed]  =  443                                                                           # Any old number. We don't currently use these
          
            if DEBUG>888:
              print ( f"{WHITE}GENERATE:       INFO:                  fnames_new = {MIKADO}{fnames_new}{RESET}",  flush=True )
              time.sleep(.4)  

            global_rna_files_processed+=1

            if DEBUG>9:
              print ( f"{WHITE}GENERATE:       INFO: global_rna_files_processed = {MIKADO}{global_rna_files_processed}{RESET}",  flush=True )
              print ( f"{DIM_WHITE}GENERATE:       INFO: n_samples                  = {CYAN}{n_samples}{RESET}",               flush=True )

          





  # (5) Summary stats

  if DEBUG>2:
    if ( input_mode=='image' ):
      print ( f"GENERATE:       INFO:  user defined tiles per sample      = {MIKADO}{n_tiles}{RESET}" )
      print ( f"GENERATE:       INFO:  total number of tiles processed    = {MIKADO}{tiles_processed}{RESET}")     
      print ( "GENERATE:       INFO:    (Numpy version of) images_new-----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( images_new     )))
      print ( "GENERATE:       INFO:    (Numpy version of) fnames_new  (dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( fnames_new     ))) 
      print ( "GENERATE:       INFO:    (Numpy version of) img_labels_new (dummy data) -----------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( img_labels_new ))) 
  
    if ( input_mode=='rna' )  | (  input_mode=='image_rna' ):  
      if use_autoencoder_output=='False':
        print ( "GENERATE:       INFO:    (Numpy version of) genes_new -----------------------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( genes_new      )))
        print ( "GENERATE:       INFO:    (Numpy version of) gnames_new ( dummy data) --------------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( gnames_new     )))   
        print ( "GENERATE:       INFO:    (Numpy version of) rna_labels_new (dummy data) -----------------------------------------------------------------------------------size in  bytes = {:,}".format(sys.getsizeof( rna_labels_new ))) 





  # (6) convert everything into Torch style tensors

  if ( input_mode=='image' ):
    images_new   = torch.Tensor( images_new )
    fnames_new   = torch.Tensor( fnames_new ).long()
    fnames_new.requires_grad_( False )
    img_labels_new  = torch.Tensor( img_labels_new ).long()                                                # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
    img_labels_new.requires_grad_( False )                                                                 # labels aren't allowed gradients
    print( "GENERATE:       INFO:  finished converting image data and labels from numpy array to Torch tensor")


  if ( input_mode=='rna' )  | ( input_mode=='image_rna'):
    
    if use_autoencoder_output=='True':                                                                     # then we already have them in Torch format, in the ae feature file, which we now load

      fpath = '%s/ae_output_features.pt' % args.log_dir
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}GENERATE:       INFO:  about to load autoencoder generated feature file from {MAGENTA}{fpath}{RESET}", flush=True )
      try:
        genes_new    = torch.load( fpath )
        genes_new    = genes_new.unsqueeze(1)                                                              # add a dimension to make it compatible with existing (non-autoencoder) code
        n_genes      = genes_new.shape[2]                                                                  # i.e. number of gene-like-features from the dimensionality reduced output of the autoencoder
        args.n_genes = n_genes
        if DEBUG>0:
          print ( f"GENERATE:       INFO:    genes_new.size         = {MIKADO}{genes_new.size()}{RESET}"      ) 
          print ( f"GENERATE:       INFO:    n_genes  (determined)  = {MIKADO}{n_genes}{RESET}"               )               
#        genes_new = np.ones( ( n_samples, 1, n_genes                 ), dtype=np.float64 )
#        if DEBUG>0:
#          print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")        
        if DEBUG>0:   
          print( f"{BRIGHT_GREEN}GENERATE:       INFO:  autoencoder feature file successfully loaded{RESET}" )          
      except Exception as e:
        print ( f"{RED}GENERATE:       INFO:  could now load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
        if DEBUG>0:
          print ( f"{RED}GENERATE:       INFO:  the exception was: {CYAN}'{e}'{RESET}" )
        sys.exit(0)

    else:          
      genes_new    = torch.Tensor( genes_new   )
    
    gnames_new   = torch.Tensor( gnames_new  ) 
    gnames_new.requires_grad_( False )        
    rna_labels_new  = torch.Tensor(rna_labels_new).long()                                                  # have to explicity cast as long as torch. Tensor does not automatically pick up type from the numpy array. 
    rna_labels_new.requires_grad_( False )                                                                 # labels aren't allowed gradients

  if DEBUG>8:
    print( "GENERATE:       INFO:  finished converting rna   data and labels     from numpy array to Torch tensor")


  if DEBUG>8:
    if ( input_mode=='image' ):
      print ( f"GENERATE:       INFO:    Torch size of images_new      =  (~tiles, rgb, height, width) {MIKADO}{images_new.size()}{RESET}"    )
      print ( f"GENERATE:       INFO:    Torch size of fnames_new      =  (~tiles)                     {MIKADO}{fnames_new.size()}{RESET}"    )
      print ( f"GENERATE:       INFO:    Torch size of img_labels_new  =  (~tiles)                     {MIKADO}{img_labels_new.size()}{RESET}" )
  
    if ( input_mode=='rna' )  |  ( input_mode=='image_rna' ):  
      print ( f"GENERATE:       INFO:    Torch size of genes_new       =  (~samples)                   {MIKADO}{genes_new.size()}{RESET}"      )
      print ( f"GENERATE:       INFO:    Torch size of gnames_new      =  (~samples)                   {MIKADO}{gnames_new.size()}{RESET}"     )
      print ( f"GENERATE:       INFO:    Torch size of rna_labels_new  =  (~samples)                   {MIKADO}{rna_labels_new.size()}{RESET}" )

  if DEBUG>88:
    if ( input_mode=='rna' )  |  ( input_mode=='image_rna' ):  
      print ( f"GENERATE:       INFO:    fnames_new                    =                               {MIKADO}{fnames_new}{RESET}"    )

  if DEBUG>6:
    if ( input_mode=='rna' )  |  ( input_mode=='image_rna' ):  
      print ( f"GENERATE:       INFO:    rna_labels_new                =                             \n{MIKADO}{rna_labels_new.numpy()}{RESET}"    )  
  
  
  # (7) save as torch '.pth' file for subsequent loading by dataset function

  if DEBUG>8:  
    print( f"GENERATE:       INFO:    {PINK}now saving to Torch dictionary (this takes a little time){RESET}")

  if input_mode=='image':
    torch.save({
        'images':     images_new,
        'fnames':     fnames_new,
        'img_labels': img_labels_new,
    }, '%s/train.pth' % cfg.ROOT_DIR)
    
  elif ( input_mode=='rna' ) | ( input_mode=='image_rna' ):
    torch.save({
        'genes':      genes_new,
        'fnames':     fnames_new,
        'gnames':     gnames_new, 
        'rna_labels': rna_labels_new,           
    }, '%s/train.pth' % cfg.ROOT_DIR)


  print( f"GENERATE:       INFO:    finished saving Torch dictionary to {MAGENTA}{cfg.ROOT_DIR}/train.pth{RESET}" )

  if ( input_mode=='rna' )  | (input_mode=='image_rna') :  
    return ( n_genes )
  else:
    return ( 0 )




#----------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
#----------------------------------------------------------------------------------------------------------

def check_mapping_file ( args, case ):

  if args.mapping_file_name=='none':
    return True
  else:                                                                                                    # a custom mapping file is active
  # then look inside the custom mapping file to see if this case exists
    exists = True
    return ( exists )


#----------------------------------------------------------------------------------------------------------
def process_rna_file ( args, genes_new, rna_labels_new, fnames_new, rna_file_link_id, gnames_new, global_rna_files_processed, rna_file, label_file, gene_data_norm, gene_data_transform, use_autoencoder_output ):


  # set up the pytorch array (using the paramaters that were passed in: genes_new, rna_labels_new, gnames_new

  if DEBUG>8:
    print ( f"{DIM_WHITE}GENERATE:       INFO:  file                         = {BLEU}{f}{RESET}", flush=True )
  
  if use_autoencoder_output=='False':                                                                      # Skip gene processing (but do labels and gnames) if we're using autoencoder output. We'll LATER load and use ae output file as genes_new rather than process raw rna-seq data 

    try:
      rna = np.load( rna_file )
      if DEBUG>9:
        print ( f"GENERATE:       INFO:         rna.shape       =  '{MIKADO}{rna.shape}{RESET}' "      )
        print ( f"GENERATE:       INFO:         genes_new.shape =  '{MIKADO}{genes_new.shape}{RESET}' ")
      if DEBUG>999:
        print ( f"GENERATE:       INFO:         rna             =  '{rna}' "            )
        print ( f"GENERATE:       INFO:         genes_new       =  '{genes_new}' "      )
    except Exception as e:
      print ( f"{RED}GENERATE:       FATAL: {e} ... halting now [118]{RESET}" )
      sys.exit(0)


    if DEBUG>999:  
      print( f"GENERATE:       INFO:                     rna = {CYAN}{rna}{RESET}" )              
    
    rna[np.abs(rna) < 1] = 0                                                                               # set all the values lower than 1 to be 0
    
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
      print( f"{RED}GENERATE:      FATAL:        no such gene data transformation as: {gene_data_transform} ... halting now[184]{RESET}" )
      sys.exit(0) 

    if gene_data_norm=='NONE':
      normalized_rna =  transformed_rna
    elif gene_data_norm=='JUST_SCALE':
      normalized_rna = transformed_rna / np.std(transformed_rna)   
    elif gene_data_norm=='GAUSSIAN':
      normalized_rna = ( transformed_rna - np.mean(transformed_rna) ) / np.std(transformed_rna)                                             
    else:
      print( f"{RED}GENERATE:      FATAL:        no such gene normalization mode as: {gene_data_norm} ... halting now[378]{RESET}" )  
      sys.exit(0)       

    genes_new [global_rna_files_processed] =  np.transpose(normalized_rna)               
      
    if DEBUG>99:
      print ( f"GENERATE:       INFO:         rna.shape       =  '{CYAN}{rna.shape}{RESET}' "      )
      print ( f"GENERATE:       INFO:         genes_new.shape =  '{CYAN}{genes_new.shape}{RESET}' ")
    if DEBUG>999:
      print ( f"GENERATE:       INFO:         rna             =  \n'{CYAN}{np.transpose(rna[1,:])}{RESET}' "      )
      print ( f"GENERATE:       INFO:         genes_new [{global_rna_files_processed}] =  '{CYAN}{genes_new[global_rna_files_processed]}{RESET}' ")                       
    
  try:
    label = np.load( label_file)
    if DEBUG>99:
      print ( "GENERATE:       INFO:         label.shape =  \"{:}\"".format(  label.shape) )
      print ( "GENERATE:       INFO:         label       =  \"{:}\"".format(  label      ) )
    if DEBUG>2:
      print ( f"{label[0]},", end='', flush=True )
  except Exception as e:
    print ( f"{RED}TRAINLENEJ:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}TRAINLENEJ:     FATAL:  explanation: expected a numpy file named {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} containing the current sample's class number in this location: {MAGENTA}{label_file}{RESET}{RED}{RESET}" )
    print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 1: probably no {MAGENTA}{args.class_numpy_file_name}{RESET}{RED} files exist. Use '{CYAN}./do_all.sh rna <cancer code> {RESET}{RED}' to regenerate them{RESET}" ) 
    print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 2: if that doesn't work, use '{CYAN}./do_all.sh rna <cancer code> regen{RESET}{RED}'. This will regenerate every file in the working dataset from respective sources (note: it can take a long time so try remedy one first){RESET}" )                                    
    print ( f"{RED}TRAINLENEJ:     FATAL:  remedy 3: this error can also occur if the user specified mapping file (currently filename: '{CYAN}{args.mapping_file_name}{RESET}{RED}') doesn't exist in '{CYAN}{args.global_data}{RESET}{RED}', because without it, no class files can be generated'{RESET}" )                                    
    print ( f"{RED}TRAINLENEJ:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)     
    
  rna_labels_new[global_rna_files_processed] =  label[0]
  
  if DEBUG>777:
    print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_labels_new[{CYAN}{global_rna_files_processed}{RESET}]  = {CYAN}{label[0]}{RESET}", flush=True )


  fnames_new [global_rna_files_processed  ]  =  rna_file_link_id                                                            # link to folder from which that this rna sample belongs to - passed in as a parameter

  if DEBUG>888:
    print ( f"{DIM_WHITE}GENERATE:       INFO:        rna_file_link_id = {MIKADO}{rna_file_link_id}{RESET}",                          flush=True )
    print ( f"{DIM_WHITE}GENERATE:       INFO:        fnames_new[{MIKADO}{global_rna_files_processed}{RESET}{DIM_WHITE}]    = {MIKADO}{fnames_new [global_rna_files_processed  ]}{RESET}", flush=True )
  
  
  gnames_new [global_rna_files_processed]  =  443                                                                           # Any old number. We don't currently use these

  result = SUCCESS
  
  return ( result )




#----------------------------------------------------------------------------------------------------------
def process_image_files ( args, dir_path, dirs, files, images_new, img_labels_new, fnames_new, n_tiles, global_tiles_processed ):

  # 1  find the SVS file in each directory then  make and store an integer reference to it (which will include the case id) so for later use when we are displaying tiles that belong to it

  for f in sorted (files):                                                                                 # examine every file in the current directory

    if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
      
      svs_file_link_id   = random.randint(1000000, 9999999)                                                # generate random string to use for the softlink to the file name (can't have strings in tensors)
      svs_file_link_name = f"{svs_file_link_id:d}"

      fqsn = f"{dir_path}/entire_patch.npy"
      fqln = f"{args.data_dir}/{svs_file_link_name}.fqln"                                                  # name for the link
      try:
        os.symlink( fqsn, fqln)                                                                            # make the link
      except Exception as e:
        if DEBUG>2:
          print ( f"{ORANGE}GENERATE:       NOTE:  Link already exists{RESET}" )
        else:
          pass

      if DEBUG>2:
        print (f"GENERATE:       INFO:  currently processing {MIKADO}{args.n_tiles[0]}{RESET} tiles from slide '{MAGENTA}{fqsn}{RESET}'" )


      if DEBUG>2:
        print( f"GENERATE:       INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}" )
        print( f"GENERATE:       INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'" )                
          
      if DEBUG>2:
        print( f"GENERATE:       INFO:                    svs_file_link_id =  {MAGENTA}{svs_file_link_id}{RESET}" )
        print( f"GENERATE:       INFO:                  svs_file_link_name = '{MAGENTA}{svs_file_link_name}{RESET}'" )
        print (f"GENERATE:       INFO:  fully qualified file name of slide = '{MAGENTA}{fqsn}{RESET}'" )
        print (f"GENERATE:       INFO:                            data_dir = '{MAGENTA}{data_dir}{RESET}'" )              
        print (f"GENERATE:       INFO:    symlink for referencing the FQSN = '{MAGENTA}{fqln}{RESET}'" )


  # 2  set up the pytorch array (using the parameters that were passed in: images_new, img_labels_new, fnames_new

  tile_extension  = "png"
  tiles_processed = 0

  for f in sorted( files ):                                                                                # examine every file in the current directory
           
    if DEBUG>999:
      print( f"GENERATE:       INFO:               files                  = {MAGENTA}{files}{RESET}"      )
  
    image_file    = os.path.join(dir_path, f)
    label_file    = os.path.join(dir_path, args.class_numpy_file_name)


    if DEBUG>2:  
      print( f"GENERATE:       INFO:               image_file    = {MAGENTA}{image_file}{MAGENTA}", flush=True   )
      print( f"GENERATE:       INFO:               label_file    = {MAGENTA}{label_file}{RESET}",   flush=True   )

    
    if ( f.endswith('.' + tile_extension ) & (not ( 'mask' in f ) ) & (not ( 'ized' in f ) )   ):          # because there may be other png files in each image folder besides the tile image files

      try:
        img = cv2.imread( image_file )
        if DEBUG>99:
          print ( f"GENERATE:      label       =  {MIKADO}{image_file}{RESET}"    )
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL: when processing: '{image_file}'{RESET}", flush=True)    
        print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)
        sys.exit(0)    


      try:
        images_new [global_tiles_processed,:] =  np.moveaxis(img, -1,0)                                    # add it to the images array
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL:  [3322] reported error was: '{e}'{RESET}", flush=True )
        print ( f"{RED}GENERATE:                      Explanation: The dimensions of the array reserved for tiles is  {MIKADO}{images_new [global_tiles_processed].shape}{RESET}{RED}; whereas the tile dimensions are: {MIKADO}{np.moveaxis(img, -1,0).shape}{RESET}", flush=True )                 
        print ( f"{RED}GENERATE:                      {RED}Did you change the tile size without regenerating the tiles? {RESET}", flush=True )
        print ( f"{RED}GENERATE:                      {RED}Either run'{CYAN}./do_all.sh <cancer type code> image{RESET}{RED}' to generate {MIKADO}{images_new [global_tiles_processed].shape[1]}x{images_new [global_tiles_processed].shape[1]}{RESET}{RED} tiles, or else change '{CYAN}TILE_SIZE{RESET}{RED}' to {MIKADO}{np.moveaxis(img, -1,0).shape[1]}{RESET}", flush=True )                 
        print ( f"{RED}GENERATE:                      {RED}Halting now{RESET}", flush=True )                 
        sys.exit(0)

      try:                                                                                                 # every tile has an associated label - the same label for every tile image in the directory
        label = np.load( label_file )
        if DEBUG>99:
          print ( f"GENERATE:      label.shape =  {MIKADO}{label.shape}{RESET}"   )
          print ( f"GENERATE:      label       =  {MIKADO}{label[0]}{RESET}"      )
        if DEBUG>2:
          print ( f"{label[0]},", end='', flush=True )
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL: when processing: '{label_file}'{RESET}", flush=True)        
        print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)
        sys.exit(0)
        
      img_labels_new[global_tiles_processed] =  label[0]                                                   # add it to the labels array
      #img_labels_new[global_tiles_processed] =  random.randint(0,5)                                       # swap truth labels to random numbers for testing purposes

      if DEBUG>77:  
        print( f"GENERATE:       INFO:               label                  = {MIKADO}{label[0]:<8d}{RESET}", flush=True   )
        

      fnames_new [global_tiles_processed]  =  svs_file_link_id                                             # link to filename of the slide from which this tile was extracted - see above

      if DEBUG>99:
          print( f"GENERATE:       INFO: symlink for tile (fnames_new [{BLUE}{global_tiles_processed:3d}{RESET}]) = {BLUE}{fnames_new [global_tiles_processed]}{RESET}" )
      

      if DEBUG>9:
        print ( "=" *180)
        print ( "GENERATE:       INFO:          tile {:} for this image:".format( global_tiles_processed+1))
        print ( "GENERATE:       INFO:            images_new[{:}].shape = {:}".format( global_tiles_processed,  images_new[global_tiles_processed].shape))
        print ( "GENERATE:       INFO:                size in bytes = {:,}".format(images_new[global_tiles_processed].size * images_new[global_tiles_processed].itemsize))  
      if DEBUG>99:
        print ( "GENERATE:       INFO:                value = \n{:}".format(images_new[global_tiles_processed]))

      the_class=img_labels_new[global_tiles_processed]
      if the_class>3000:
          print ( f"{RED}GENERATE:       FATAL: Ludicrously large class value detected (class={MIKADO}{the_class}{RESET}{RED}) for tile '{MAGENTA}{image_file}{RESET}" )
          print ( f"{RED}GENERATE:       FATAL: hanting now [1718]{RESET}" )
          sys.exit(0)
          
      if DEBUG>9:
        size_in_bytes=img_labels_new[global_tiles_processed].size * img_labels_new[global_tiles_processed].itemsize
        print ( f"GENERATE:       INFO:        for img_labels_new[{global_tiles_processed}]; class={the_class}" )

      if DEBUG>99:
        print ( "GENERATE:       INFO:            fnames_new[{:}]".format( global_tiles_processed ) )
        print ( "GENERATE:       INFO:                size in  bytes = {:,}".format( fnames_new[global_tiles_processed].size * fnames_new[global_tiles_processed].itemsize))
        print ( "GENERATE:       INFO:                value = {:}".format( fnames_new[global_tiles_processed] ) )
       
      global_tiles_processed+=1
      
      tiles_processed+=1
      if tiles_processed==n_tiles:
        break
      
    else:
      if DEBUG>1:
        print( f"GENERATE:       INFO:          other file = {MIKADO}{image_file}{RESET}".format(  ) )
        
    
  if DEBUG>7:
    print( f"GENERATE:       INFO:                              tiles processed in in directory: '{MAGENTA}{dir_path}{RESET}' = {ARYLIDE}{tiles_processed:<8d}{RESET}",        flush=True       )   
    
  if  ( args.just_test=='False' ):
    if (tiles_processed!=n_tiles) & (tiles_processed!=0):
      print( f"{RED}GENERATE:       INFO:     tiles processed in directory: '{MAGENTA}{dir_path}{RESET}' = \r\033[150C{MIKADO}{tiles_processed:<8d}{RESET}{RED}\r\033[180C<<<<<<<<<<<< anomoly {RESET}", flush=True  )       
      time.sleep(4)
  
  return global_tiles_processed
