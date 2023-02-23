"""
open each rna file in data_dir; extract 'tcga_rna_seq_metric'; convert to numpy array; save numpy array as a new file with the same name but .npy extension 

"""

import os
import sys
import codecs
import random
import fnmatch
import argparse
import numpy  as np
import pandas as pd

from constants  import *

DEBUG   = 1

a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

pd.set_option('display.max_rows',     999 )
pd.set_option('display.max_columns',  99 )
pd.set_option('display.width',        999 )
pd.set_option('display.max_colwidth', 999 ) 
pd.set_option('display.float_format', lambda x: '%6.2f' % x)    

#====================================================================================================================================================
def process_rna_seq( args ):

  files_reviewed_count   = 0
  cumulative_rna_results = 0
  
  data_dir                   = args.data_dir
  tcga_rna_seq_file_suffix   = args.tcga_rna_seq_file_suffix
  tcga_rna_seq_metric        = args.tcga_rna_seq_metric
  tcga_rna_seq_start_row     = args.tcga_rna_seq_start_row
  rna_file_reduced_suffix    = args.rna_file_reduced_suffix
  rna_numpy_filename         = args.rna_numpy_filename
  use_unfiltered_data        = args.use_unfiltered_data
  skip_rna_preprocessing     = args.skip_rna_preprocessing
  
  if  skip_rna_preprocessing == True:
    print( f"{ORANGE}PROCESS_RNA_SEQ INFO: '{CYAN}skip_rna_preprocessing{RESET}{ORANGE}' flag = {MIKADO}{skip_rna_preprocessing}{RESET}{ORANGE}. Processing of raw RNA-Seq files will NOT be performed, and '{MAGENTA}{rna_numpy_filename}{RESET}{ORANGE}' files will NOT be generated. {RESET}" )
    print( f"{ORANGE}PROCESS_RNA_SEQ INFO: This may be intentional on your part: the required files may alreay exist, and you may be using this flag to avoid repeatedly generating these files again. {RESET}" )
    return
  
  if (DEBUG>99):
    print ( f"PROCESS_RNA_SEQ:        INFO: data directory                        = {MAGENTA}'{data_dir}'{RESET}",                  flush=True )
    print ( f"PROCESS_RNA_SEQ:        INFO: RNA-Seq file suffix                   = {MAGENTA}{rna_file_reduced_suffix}{RESET}",     flush=True )
    print ( f"PROCESS_RNA_SEQ:        INFO: RNA-Seq class (subtype) column number = {MAGENTA}{tcga_rna_seq_metric}{RESET}",              flush=True )
    print ( f"PROCESS_RNA_SEQ:        INFO: RNA-Seq file name                     = {MAGENTA}'{rna_numpy_filename};{RESET}",        flush=True )

  if use_unfiltered_data == True: 
    suffix = f"{tcga_rna_seq_file_suffix}"                                                                 # no filtering was performed in the previous step. Look for file name without '_reduced' suffix   
    sep='\t'
  else:
    suffix = f"*{rna_file_reduced_suffix}"                                                                 # file ending with "_reduced", generated by 'process_rna_exp.py' in the previous step
    sep='\t'
    

  if (DEBUG>0):
    print ( f"PROCESS_RNA_SEQ:        INFO: will look recursively under:            {MAGENTA}'{data_dir}'{RESET} for files with the TCGA RNA-Seq suffix, that is {BB}{suffix}{RESET}",  flush=True ) 
           
  walker = os.walk(data_dir)
  for base_dir, __, files in walker:
    
    for f in files:
    
      current_file    = os.path.join( base_dir, f)
  
      if (DEBUG>10):
        print ( f"\n\nPROCESS_RNA_SEQ:        INFO: (current_file)                      = {BB}{current_file}{RESET}",                     flush=True )  
        print ( f"PROCESS_RNA_SEQ:        INFO: suffix                              = {BB}{suffix}{RESET}",                           flush=True )
  
      if fnmatch.fnmatch( f, suffix  ):                                                                    # if found ...
   
        rna_results_file_found   =1
        cumulative_rna_results  +=1  
        
        if (DEBUG>2): 
          print ( f"PROCESS_RNA_SEQ:        INFO: (match !)                          {BB}{current_file}{RESET}    \r\033[220Ccumulative match count       = {ARYLIDE}{cumulative_rna_results}{RESET}",  flush=True )
                  
        rna_numpy_file_fqn = os.path.join( base_dir, rna_numpy_filename )                                      # rna.npy
        
        if (DEBUG>10):
          print ( f"PROCESS_RNA_SEQ:        INFO: tcga_rna_seq_metric          = {BB}{tcga_rna_seq_metric}{RESET}",  flush=True )  
        
        rna_expression_column = pd.read_csv( current_file, usecols=[ tcga_rna_seq_metric ], skiprows=tcga_rna_seq_start_row, sep=sep, header=None )

        if (DEBUG>10):      
          print ( f"PROCESS_RNA_SEQ:        INFO: rna_expression_column (pandas df)   ={BB}{rna_expression_column}{RESET}",  flush=True )  
        
        if DEBUG>2:
          v = np.transpose( rna_expression_column[0:20])
          print ( f"PROCESS_RNA_SEQ: median = {MIKADO}{np.median(v):6.1f}{RESET}" )
          pd.set_option('display.float_format', lambda x: '%12.1f' % x)
          print ( f"PROCESS_RNA_SEQ: rna_expression_column (first 20 entries) = \n{GREEN if np.median(v) > 200 else RED}{v}{RESET}" )
        
        rna = rna_expression_column.to_numpy()

        if DEBUG>999:
          print ( f"PROCESS_RNA_SEQ:        INFO:  rna.shape                          = {CYAN}{rna.shape}{RESET}",                      flush=True      )
        if DEBUG>999:
          print ( f"PROCESS_RNA_SEQ:        INFO:  rna                                = {CYAN}{np.transpose(rna[0:50])}{RESET}",        flush=True      )
        if (DEBUG>10):              
          print ( f"PROCESS_RNA_SEQ:        INFO: rna_numpy_file_fqn (will be)        = {BB}{rna_numpy_file_fqn}{RESET}",                flush=True     )  
        
        np.save( rna_numpy_file_fqn, rna )                                                                 # rna.npy

      files_reviewed_count += 1

      if (DEBUG>0):
        if files_reviewed_count % 50==0:
          print ( f"PROCESS_RNA_SEQ:        INFO: {MIKADO}{files_reviewed_count}{RESET} files reviewed, of which {MIKADO}{cumulative_rna_results}{RESET} TCGA format RNA-Seq files found. Numpy versions made and stored for these {MIKADO}{cumulative_rna_results}{RESET} RNA-Seq files{RESET}",  flush=True )
          print ( "\033[2A",  flush=True )

  if (DEBUG>0):                                                                                          # this will show the final count
    print ( f"PROCESS_RNA_SEQ:        INFO: {MIKADO}{files_reviewed_count}{RESET} files reviewed, of which {MIKADO}{cumulative_rna_results}{RESET} TCGA format RNA-Seq files found. Numpy versions made and stored for these {MIKADO}{cumulative_rna_results}{RESET} RNA-Seq files{RESET}",  flush=True )
    print ( "\033[2A",  flush=True )

  if (DEBUG>0):
    print ( "\033[1B",  flush=True )

