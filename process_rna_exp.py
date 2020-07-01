"""
open each rna results file within the patches directory; extract the rna-seq "scaled estimate" column; convert to numpy array; save numpy array as a new file with the same name but .npy rna_file_reduced_suffix 

"""

import os
import sys
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

DEBUG=1

RESET="\033[m"
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

#====================================================================================================================================================
def main(args):

  cumulative_rna_results = 0
  
  data_dir                 = args.data_dir
  rna_file_reduced_suffix  = args.rna_file_reduced_suffix
  rna_exp_column           = args.rna_exp_column
  rna_numpy_filename       = args.rna_numpy_filename
  
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
  
      pattern = f"*{rna_file_reduced_suffix}"
      
      # Handle RNA data
      if fnmatch.fnmatch( f, pattern  ):
   
        rna_results_file_found   =1
        cumulative_rna_results  +=1  
        
        if (DEBUG>0): 
          print ( "PROCESS_RNA_EXP:        INFO: (match !)                     {:}{:}{:}{:}    cumulative match count = {:}{:}".format( BB, current_file, RESET, BB, cumulative_rna_results, RESET ),  flush=True )
                  
        rna_npy_file          = os.path.join( root, rna_numpy_filename )
        
        if (DEBUG>99): 
          print ( "PROCESS_RNA_EXP:        INFO: (rna_npy_file)                  = {:}{:}{:}".format( BB, rna_npy_file, RESET ),  flush=True )  

        rna_expression_column = pd.read_csv(current_file, sep=',', usecols=[rna_exp_column])
        
        if DEBUG>99:
          print ( "PROCESS_RNA_EXP: rna_expression_column as Pandas object = \n\033[35m{:}\033[m".format(rna_expression_column[0:50]))
        
        rna = rna_expression_column.to_numpy()
    
        if DEBUG>99:
          print ( "PROCESS_RNA_EXP: rna_expression_column as Numpy array   = \n\033[35m{:}\033[m".format(rna[0:50]))
        
        np.save(rna_npy_file, rna)
      
#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                type=str, default="/home/peter/git/pipeline/dataset")
  p.add_argument('--rna_file_reduced_suffix', type=str, default='_reduced')
  p.add_argument('--rna_exp_column',          type=int, default=1)
  p.add_argument('--rna_numpy_filename',      type=str, default="rna.npy") 
  
  args, _ = p.parse_known_args()

  main(args)
      
