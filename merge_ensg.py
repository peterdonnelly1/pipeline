import os
import re
import cv2
import sys
import time
#import cuda
import cupy
import shutil
import argparse
import torch
import fnmatch
import random
import numpy as np  
import pandas as pd
from tabulate import tabulate

np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

pd.set_option('display.max_rows',     50 )
pd.set_option('display.max_columns',  13 )
pd.set_option('display.width',       300 )
pd.set_option('display.max_colwidth', 99 )  

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1


def main( args ):

  pool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
  cupy.cuda.set_allocator(pool.malloc)

  base_dir                    = args.base_dir
  data_dir                    = args.data_dir


  # convert to pandas dataframe, then pickle and save for possible use with analyse_data
  

  ensg_reference_file_name = f"{data_dir}/ENSG_UCSC_biomart_ENS_id_to_gene_name_table"
  ensg_reference_file_save_name = f"{data_dir}/ENSG_UCSC_biomart_ENS_id_to_gene_name_table_normalized"  
  if DEBUG>0:  
    print ( f"P_C_GENERATE:       INFO:      loading ensg_reference_file_name (containing ENSG-> gene name mapping) from {MAGENTA}{ensg_reference_file_name}{RESET}", flush=True )      
  df_map = pd.read_csv( ensg_reference_file_name, sep='\t' )
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      removing duplicates", flush=True )                           # there are a lot of duplicates in the table downloaded from UCSC
  df_map.drop_duplicates( keep='first', inplace=True )
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      adding column headers", flush=True )
  df_map.columns=['ENSG_id', 'gene_name']
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      sorting", flush=True )  
  df_map = df_map.sort_values( by='ENSG_id', ascending=True)
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      resetting index", flush=True )
  df_map = df_map.reset_index(drop=True)    
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      saving normalized version of df_map to {MAGENTA}{ensg_reference_file_save_name}{RESET}", flush=True )  
  df_map.to_csv( ensg_reference_file_save_name, sep='\t' )  
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      pandas description of df_map: \n{CYAN}{df_map.describe}{RESET}", flush=True )  
  if DEBUG>99:
    print(tabulate(df_map, tablefmt='psql'))

  gene_name_reference_file_name = f"{data_dir}/ENSG_reference"
  gene_name_reference_file_save_name = f"{data_dir}/ENSG_reference_normalized"
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      loading gene_name_reference      (containing just the TCGA ENSG IDs) from {MAGENTA}{gene_name_reference_file_name}{RESET}", flush=True )
  df_tcga = pd.read_csv( gene_name_reference_file_name, sep='\t')
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      removing duplicates", flush=True )
  df_tcga.drop_duplicates( keep='first', inplace=True )                                                    
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      stripping version numbers from ENSG IDs", flush=True )
  for el in range (0, df_tcga.shape[0]):                                                                   # strip the version number from the Ensembl ID (i.e. retain JUST the leading 11 non version number digits)
    if DEBUG>99:
      print ( df_tcga.iloc[el,0] ) 
    mod = re.sub( '(^ENS[A-Z][0-9]{11}).*$', r'\1', df_tcga.iloc[el,0] )
    df_tcga.iloc[el,0] = mod
    if DEBUG>99:
      print ( f"{PINK}{mod}{RESET}" )
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      adding column headers", flush=True )
  df_tcga.columns=['ENSG_id']
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      sorting", flush=True )  
  df_tcga = df_tcga.sort_values( by='ENSG_id', ascending=True)
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      resetting index", flush=True )
  df_tcga = df_tcga.reset_index(drop=True)
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      saving normalized version to {MAGENTA}{gene_name_reference_file_save_name}{RESET}", flush=True )  
  df_tcga.to_csv( gene_name_reference_file_save_name, sep='\t' )    
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      pandas description of df_tcga: \n{CYAN}{df_tcga.describe}{RESET}", flush=True )
  if DEBUG>99:
    print(tabulate(df_tcga, tablefmt='psql'))
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      merging df_tcga with df_map (the latter has gene names)", flush=True )  
  df_merge = df_tcga.merge(df_map, how='left', indicator='True' )
  if DEBUG>99:
    print(tabulate(df_merge, tablefmt='psql'))

  ENSG_reference_merged_file_save_name = f"{data_dir}/ENSG_reference_merged"
  if DEBUG>0:
    print ( f"P_C_GENERATE:       INFO:      saving merged version to {MAGENTA}{ENSG_reference_merged_file_save_name}{RESET}", flush=True )  
  df_merge.to_csv( ENSG_reference_merged_file_save_name, sep='\t', index=False )  


# ------------------------------------------------------------------------------
if __name__ == '__main__':

  p = argparse.ArgumentParser()

  p.add_argument('--log_dir',                        type=str,   default='data/dlbcl_image/logs')
  p.add_argument('--base_dir',                       type=str,   default='/home/peter/git/pipeline')
  p.add_argument('--data_dir',                       type=str,   default='/home/peter/git/pipeline/dataset')     
  p.add_argument('--save_model_every',               type=int,   default=10)                                     
  p.add_argument('--nn_mode',                        type=str,   default='dlbcl_image')
  p.add_argument('--min_tile_sd',                    type=float, default=3)                                      
  p.add_argument('--greyness',                       type=int,   default=0)                                      
  p.add_argument('--class_colours',      nargs="*"                                  )    
      
  args, _ = p.parse_known_args()

  main(args)