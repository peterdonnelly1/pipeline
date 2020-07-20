"""
open each rna file in data_dir; extract 'rna_exp_column' (column 1); convert to numpy array; save numpy array as a new file with the same name but .npy extension 

"""

import os
import sys
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

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

DEBUG=100

RESET="\033[m"
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

#====================================================================================================================================================
def main(args):

  cumulative_rna_results = 0
  
  data_dir                 = args.data_dir
  rna_file_suffix          = args.rna_file_suffix  
  rna_file_reduced_suffix  = args.rna_file_reduced_suffix
  rna_exp_column           = args.rna_exp_column
  rna_numpy_filename       = args.rna_numpy_filename
  use_unfiltered_data      = args.use_unfiltered_data  
  
  if (DEBUG>0):
    print ( "PROCESS_RNA_EXP:        INFO: args.data_dir                  = {:}{:}{:}".format( BB, data_dir,                 RESET ),  flush=True )
    print ( "PROCESS_RNA_EXP:        INFO: args.rna_file_reduced_suffix   = {:}{:}{:}".format( BB, rna_file_reduced_suffix , RESET ),  flush=True )
    print ( "PROCESS_RNA_EXP:        INFO: args.rna_exp_column            = {:}{:}{:}".format( BB, rna_exp_column,           RESET ),  flush=True )
    print ( "PROCESS_RNA_EXP:        INFO: args.rna_numpy_filename        = {:}{:}{:}".format( BB, rna_numpy_filename,       RESET ),  flush=True )

  if (DEBUG>0):
    print ( "PROCESS_RNA_EXP:        INFO: will look recursively under  {:}'{:}'{:} for files that match this pattern: {:}{:}{:}".format( BB, data_dir, RESET, BB, rna_file_reduced_suffix, RESET ),  flush=True ) 
           
  walker = os.walk(data_dir)
  for root, __, files in walker:
    
    for f in files:
    
      current_file    = os.path.join( root, f)
  
      if (DEBUG>99):
        print ( "PROCESS_RNA_EXP:        INFO: (current_file)                    \033[34m{:}\033[m".format(   current_file          ),  flush=True )  
  
      if use_unfiltered_data=='False':
        pattern = f"*{rna_file_reduced_suffix}"                                                          # file ending with "_reduced", generated by 'process_rna_exp.py' in the previous step
        sep=','
      else:
        pattern = f"{rna_file_suffix}"                                                                   # no filtering was performed in the previous step. Look for file name without '_reduced' suffix   
        sep='\t'

      if (DEBUG>0):
        print ( f"PROCESS_RNA_EXP:        INFO: pattern                          = {BB}{pattern}{RESET}",  flush=True )
  
      # Handle RNA data
      if fnmatch.fnmatch( f, pattern  ):                                                                 # if found ...
   
        rna_results_file_found   =1
        cumulative_rna_results  +=1  
        
        if (DEBUG>99): 
          print ( "PROCESS_RNA_EXP:        INFO: (match !)                          {:}{:}{:}{:}    cumulative match count = {:}{:}".format( BB, current_file, RESET, BB, cumulative_rna_results, RESET ),  flush=True )
                  
        rna_npy_file          = os.path.join( root, rna_numpy_filename )                                   # rna.npy
        
        if (DEBUG>99): 
          print ( "PROCESS_RNA_EXP:        INFO: (rna_npy_file)                   = {:}{:}{:}".format( BB, rna_npy_file, RESET ),  flush=True )  

        rna_expression_column = pd.read_csv(current_file, sep=sep, usecols=[rna_exp_column])               # rna_exp_column=1
        
        if DEBUG>99:
          print ( "PROCESS_RNA_EXP: rna_expression_column as Pandas object = \n\033[35m{:}\033[m".format( np.transpose(rna_expression_column[0:50])))
        
        rna = rna_expression_column.to_numpy()
    
        if DEBUG>99:
          print ( "PROCESS_RNA_EXP: rna_expression_column as Numpy array   = \n\033[35m{:}\033[m".format(np.transpose(rna[0:50])))
        
        np.save(rna_npy_file, rna)                                                                         # rna.npy
      
#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                type=str, default="/home/peter/git/pipeline/dataset")
  p.add_argument('--rna_file_suffix',         type=str, default='*FPKM-UQ.txt')
  p.add_argument('--rna_file_reduced_suffix', type=str, default='_reduced')
  p.add_argument('--rna_exp_column',          type=int, default=1)
  p.add_argument('--rna_numpy_filename',      type=str, default="rna.npy")
  p.add_argument('--use_unfiltered_data',     type=str, default='False' ) 
  
  args, _ = p.parse_known_args()

  main(args)
      
