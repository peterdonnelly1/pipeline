"""
open each rna results file within the patches directory; extract the rna-seq "scaled estimate" column; convert to numpy array; save numpy array as a new file with the same name but .npy rna_file_suffix 

"""

import os
import sys
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

DEBUG=0

RESET="\033[m"
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

#====================================================================================================================================================
def main(args):

  cumulative_rna_results = 0
  
  data_dir               = args.data_dir
  rna_file_suffix        = args.rna_file_suffix
  rna_exp_column         = args.rna_exp_column
  rna_numpy_filename     = args.rna_numpy_filename
  
  if (DEBUG>0):
    print ( "PROCESS_RNA_SEQ:        INFO: args.data_dir               = {:}{:}{:}".format( BB, data_dir,           RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_file_suffix        = {:}{:}{:}".format( BB, rna_file_suffix ,   RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_exp_column         = {:}{:}{:}".format( BB, rna_exp_column,     RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_numpy_filename     = {:}{:}{:}".format( BB, rna_numpy_filename, RESET ),  flush=True )

  if (DEBUG>0):
    print ( "PROCESS_RNA_SEQ:        INFO: will look recursively under  {:}'{:}'{:} for files that match this pattern: {:}{:}{:}".format( BB, data_dir, RESET, BB, rna_file_suffix, RESET ),  flush=True ) 
           
  walker = os.walk(data_dir)
  for root, __, files in walker:
    for f in files:
      current_file    = os.path.join( root, f)
  
      if (DEBUG>99):
        print ( "PROCESS_RNA_SEQ:        INFO: (current_file)                    \033[34m{:}\033[m".format(   current_file          ),  flush=True )  
  
      # Handle RNA data
      if fnmatch.fnmatch( f, rna_file_suffix  ):
   
        rna_results_file_found   =1
        cumulative_rna_results  +=1  
        
        if (DEBUG>0): 
          print ( "PROCESS_RNA_SEQ:        INFO: (match !)                         {:}{:}{:}{:}    cumulative match count = {:}{:}".format( BB, current_file, RESET, BB, cumulative_rna_results, RESET ),  flush=True )
                  
        rna_npy_file          = os.path.join( root, rna_numpy_filename )
        
        if (DEBUG>0): 
          print ( "PROCESS_RNA_SEQ:        INFO: (rna_npy_file)                  = {:}{:}{:}".format( BB, rna_npy_file, RESET ),  flush=True )  

        rna_expression_column = pd.read_csv(current_file, sep='\t', usecols=[1])
        
        if DEBUG>1:
          print ( "PROCESS_RNA_SEQ: rna_expression_column as pandas object = \n\033[35m{:}\033[m".format(rna_expression_column[0:12]))
        
        rna = rna_expression_column.to_numpy()
    
        if DEBUG>1:
          print ( "PROCESS_RNA_SEQ: rna_expression_column as numpy array   = \n\033[35m{:}\033[m".format(rna[0:12]))
        
        np.save(rna_npy_file, rna)
      
#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',            type=str, default="/home/peter/git/pipeline/dataseff")
  p.add_argument('--rna_file_suffix',     type=str, default="*FPKM-UQ.txt")
  p.add_argument('--rna_exp_column',      type=str, default=1)
  p.add_argument('--rna_numpy_filename',  type=str, default="rna.npy") 
  
  args, _ = p.parse_known_args()

  main(args)
      
