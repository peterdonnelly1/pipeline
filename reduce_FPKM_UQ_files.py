"""
open each rna results file remove all rows which contain genes (ENSEMBL gene IDs) that do not correspond to TARGET's cancer genes of interest; save as file with the same name but with 'reduced' suffix

     cancer genes of interest must be contained within a file called 'target_genes_of_interest' in dataset, which
        
        must be a csv file
        may contain either valid ENSEMBL gene IDs or blank cells in any row or column (don't have to be contiguous rows or columns, or contiguous cells within rows or columns)
  

"""

import os
import re
import sys
import time
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
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
CHARTREUSE='\033[38;2;223;255;0m'
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

DEBUG   = 1

FAIL    = 0
SUCCESS = 1

a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=275)
np.set_printoptions(threshold=10000)


#====================================================================================================================================================
def main(args):
  
  data_dir                    = args.data_dir
  target_genes_reference_file = args.target_genes_reference_file
  args.rna_file_suffix        = args.rna_file_suffix
  rna_file_reduced_suffix     = args.rna_file_reduced_suffix
  rna_exp_column              = args.rna_exp_column
  use_unfiltered_data         = args.use_unfiltered_data
  remove_low_expression_genes = args.remove_low_expression_genes
  low_expression_threshold    = args.low_expression_threshold

  if remove_low_expression_genes=='True':
    print( f"{ORANGE}REDUCE_FPKM_UQ_FILES:   INFO: 'remove_low_expression_genes'  flag is set. Genes whose expression value is less than {CYAN}{low_expression_threshold}{RESET} for {BOLD}all{RESET}{ORANGE} samples will be deleted prior to any other filter being applies{RESET}" )

  if  use_unfiltered_data=='True':
    print( f"{ORANGE}REDUCE_FPKM_UQ_FILES:   INFO: '{CYAN}use_unfiltered_data{RESET}{ORANGE}' flag = {MIKADO}True{RESET}{ORANGE}. No filtering will be performed, and '{MAGENTA}_reduced{RESET}{ORANGE}' files will NOT be generated. {RESET}" )
    sys.exit(0)
  else:
    print( f"{ORANGE}REDUCE_FPKM_UQ_FILES:   INFO: '{CYAN}use_unfiltered_data{RESET}' flag = {MIKADO}True{RESET}{ORANGE}. Filtering will be performed; '{MAGENTA}_reduced{RESET}{ORANGE}' files WILL be generated. {RESET}" )    
    
  result = reduce_genes( args, target_genes_reference_file )

  if result==FAIL:
    print( f"{RED}REDUCE_FPKM_UQ_FILES:   FATAL:   reduce_genes() returned 'FAIL' ... halting now {RESET}" )
    sys.exit(0)


def reduce_genes( args, target_genes_reference_file ):
  
  cumulative_found_count = 0

  if (DEBUG>0):
    print ( f"{ORANGE}REDUCE_FPKM_UQ_FILES:   INFO: will look recursively under {MAGENTA}'{args.data_dir}'{ORANGE} for files that match this pattern: {BB}{args.rna_file_suffix}{RESET}",  flush=True ) 

  if (DEBUG>99):
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: target_genes_reference_file    = {MAGENTA}{args.target_genes_reference_file}{RESET}",  flush=True )
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: args.data_dir                  = {MAGENTA}{args.data_dir}{RESET}",                     flush=True )
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: args.rna_file_suffix           = {MAGENTA}{args.rna_file_suffix}{RESET}",              flush=True )
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: args.rna_exp_column            = {MIKADO}{args.rna_exp_column}{RESET}",               flush=True )

  # STEP 1: READ ENSEMBL FROM target_reference_file; REMOVE BLANKS; CONVERT TO NUMPY VECTOR
  
  try:
    target_genes_of_interest = pd.read_csv(target_genes_reference_file, sep='\t', na_filter=False )
  except Exception as e:
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL: {CYAN}pd.read_csv{RESET}{RED} failed when trying to open {CYAN}{args.target_genes_reference_file}{RESET}"  )
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:  The reported error was: {CYAN}{e}{RESET}" )
    print( f"{RED}REDUCE_FPKM_UQ_FILES:         FATAL:  ... halting now\n\n\n\n\n {RESET}" )
    sys.exit(0)


  if DEBUG>9999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: target_genes_of_interest as pandas object = \n\033[35m{target_genes_of_interest}\033[m" )

  np_pmcc_reference   = target_genes_of_interest.to_numpy()

  if DEBUG>9999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: target_genes_of_interest as numpy array = \n\033[35m{np_pmcc_reference}\033[m" )
  if DEBUG>999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_pmcc_reference.shape                 = \033[35m{np_pmcc_reference.shape}\033[m" )

  np_pmcc_reference_concatenated   = np.concatenate(np_pmcc_reference)

  if DEBUG>9999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_pmcc_reference_concatenated       = \n\033[35m{np_pmcc_reference_concatenated}\033[m" )
  if DEBUG>999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_pmcc_reference_concatenated.shape = \033[35m{np_pmcc_reference_concatenated.shape}\033[m" )

  np_pmcc_reference_concatenated = [i for i in np_pmcc_reference_concatenated if "ENSG" in i ]


  if DEBUG>9999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_pmcc_reference_concatenated with empty strings removed       = \n{MIKADO}{np_pmcc_reference_concatenated}{RESET}" )
  if DEBUG>0:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: {CYAN}len(np_pmcc_reference_concatenated{RESET} (with empty strings removed) = {MIKADO}{len(np_pmcc_reference_concatenated)}{RESET}" )


  # STEP 2: OPEN RNA "FPKM_UQ" RESULTS FILE; EXTRACT ROWS WHICH CORRESPOND TO TARGET CANCER GENES OF INTEREST, SAVE AS (TSV) FILE WITH SAME NAME AS ORIGINAL PLUS 'REDUCED' SUFFIX
  
  last_table_shape=np.array([0,0])
  
  walker = os.walk(args.data_dir)
  for root, __, files in walker:
    
    for f in files:
      
      current_fqn = os.path.join( root, f)
      new_f       = f"{f}{args.rna_file_reduced_suffix}"
      new_fqn     = os.path.join( root, new_f)
        
      if (DEBUG>999):
        print ( "REDUCE_FPKM_UQ_FILES:   INFO: (current_fqn)                    \033[34m{:}\033[m".format(   current_fqn          ),  flush=True )  
  
      if fnmatch.fnmatch( f, args.rna_file_suffix  ):
   
        rna_results_file_found   =1
        cumulative_found_count    +=1  
        
        if (DEBUG>99): 
          print ( f"REDUCE_FPKM_UQ_FILES:   INFO: (match !)                                     {CARRIBEAN_GREEN}{current_fqn}{RESET}    \r\033[220Ccumulative_found_count = {MIKADO}{cumulative_found_count}{RESET}",  flush=True )

        ensembl_gene_id_column = pd.read_csv(current_fqn, sep='\t', usecols=[0,1])
        np_ensemble_gene_ids   = ensembl_gene_id_column.to_numpy()

        if DEBUG>999:
          print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_ensemble_gene_ids.shape  = \033[35m{np_ensemble_gene_ids.shape}\033[m" )
          print ( f"REDUCE_FPKM_UQ_FILES:   INFO: np_ensemble_gene_ids        = \033[35m{np_ensemble_gene_ids}\033[m" )
        
        new_table = np.array([len(np_pmcc_reference_concatenated),2])      # PGD 200719 - CHECK THIS !!! PROBABLY NO LONGER VALID !!!  # assumes we can't find more than 'np_pmcc_reference_concatenated' matches, which is valid
 
        if DEBUG>9999:
          print ( f"REDUCE_FPKM_UQ_FILES:   INFO: new_table.shape             = \033[35m{new_table.shape}\033[m" )

                
        for target in np_ensemble_gene_ids:
          if DEBUG>9999:
            print ( f"REDUCE_FPKM_UQ_FILES:   INFO: main()              target = \033[35m{target}\033[m" )
          r=strip_suffix(target[0])
          if DEBUG>9999:
            print ( f"REDUCE_FPKM_UQ_FILES:   INFO: r = {r}" )
            
          if r in np_pmcc_reference_concatenated:                                                             # if the row contains one of the TARGET cancer genes of interest
            new_table = np.vstack([new_table, target])                                                     # then add the row to new_table
            new_table_shape  = new_table.shape
            last_table_shape = new_table.shape
            if not new_table_shape==last_table_shape:
              print(f"\033[31m\033[1mTILER: FATAL: size of table that will become the reduced FPKM_UQ file {new_table_shape} differs from that of the last FPKM_UQ file processed {last_table_shape}" , flush=True)
              print(f"\033[31m\033[1mTILER: FATAL: this should not happen, and will cause training to crash, so preemptively stopping now ",                                                            flush=True)
            if DEBUG>999:
              print ( f"REDUCE_FPKM_UQ_FILES:   INFO: new_table        = \033[35m{new_table}\033[m" )
              print ( f"REDUCE_FPKM_UQ_FILES:   INFO: new_table_shape  = \033[35m{new_table_shape}\033[m" )
              print ( f"REDUCE_FPKM_UQ_FILES:   INFO: last_table_shape = \033[35m{last_table_shape}\033[m" )
          else: 
            if DEBUG>9999:
              print ( "REDUCE_FPKM_UQ_FILES:   INFO: \033[31;1mthis is not one of the TARGET genes of interest -- moving on \033[m" )

        if DEBUG>9999:
          print ( f"REDUCE_FPKM_UQ_FILES:   INFO: about to save {CYAN}new_table{RESET} to name               {MAGENTA}{new_fqn}{RESET}  \r\033[220Ccumulative found count = {MIKADO}{cumulative_found_count}{RESET}"  )

        try:
          pd.DataFrame(new_table).to_csv(new_fqn, index=False, header=False, index_label=False )           # don't add the column and row labels that Pandas would otherwise add
          if DEBUG>0:
            print ( f"REDUCE_FPKM_UQ_FILES:   INFO: saving reduced file with dims \033[35m{new_table_shape}\033[m to name {MAGENTA}{new_fqn}{RESET}        \r\033[220Ccumulative found count = {MIKADO}{cumulative_found_count}{RESET}"  )
        except Exception as e:
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:   FATAL: could not save file         = {CYAN}{new_table}{RESET}"  )
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}REDUCE_FPKM_UQ_FILES:        FATAL:  The reported error was: {CYAN}{e}{RESET}" )
          print( f"{RED}REDUCE_FPKM_UQ_FILES:         FATAL:  ... halting now\n\n\n\n\n\n {RESET}" )
          sys.exit(0)



  sys.exit(0)

  return SUCCESS

#====================================================================================================================================================
def strip_suffix(s):
  
  if DEBUG>9999:
    print ( f"REDUCE_FPKM_UQ_FILES:   INFO: strip_suffix()           s = { s }", flush=True )
  
  r=re.search('^ENS[A-Z][0-9]*', s)

  if r:
    found = r.group(0)
    if DEBUG>9999:
      print ( f"REDUCE_FPKM_UQ_FILES:   INFO: strip_suffix () r.group(0) = { r.group(0) }", flush=True )
    return r.group(0)
  else:
    return 0

#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                                 type=str,   default="/home/peter/git/pipeline/dataset")
  p.add_argument('--target_genes_reference_file',              type=str,   default="/home/peter/git/pipeline/dataset/pmcc_cancer_genes_of_interest")
  p.add_argument('--rna_file_suffix',                          type=str,   default='*FPKM-UQ.txt')
  p.add_argument('--rna_file_reduced_suffix',                  type=str,   default='_reduced')
  p.add_argument('--rna_exp_column',                           type=int,   default=1)
  p.add_argument('--use_unfiltered_data',                      type=str,   default='False' )  
  p.add_argument('--remove_low_expression_genes',              type=str,   default='False' ) 
  p.add_argument('--low_expression_threshold',                 type=float, default='0.0' )   
  
  
  args, _ = p.parse_known_args()

  main(args)
      
