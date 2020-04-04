"""
open each rna results file within the patches directory; remove all rows which contain genes (ENSEMBL gene IDs) that do not coorespond to PMCC's cancer genes of interest

   PMCC's cancer genes of interested must be contained within a file called 'pmcc_genes_of_interest' in dataset, which
        
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

DEBUG=1

RESET="\033[m"
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=275)
np.set_printoptions(threshold=10000)


#====================================================================================================================================================
def main(args):

  cumulative_found_count = 0
  
  data_dir                    = args.data_dir
  pmcc_genes_reference_file   = args.pmcc_genes_reference_file
  rna_file_suffix             = args.rna_file_suffix
  rna_file_reduced_suffix     = args.rna_file_reduced_suffix
  rna_exp_column              = args.rna_exp_column
  
  if (DEBUG>99):
    print ( "PROCESS_RNA_SEQ:        INFO: args.data_dir               = {:}{:}{:}".format( BB, data_dir,           RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_file_suffix        = {:}{:}{:}".format( BB, rna_file_suffix ,   RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_exp_column         = {:}{:}{:}".format( BB, rna_exp_column,     RESET ),  flush=True )

  if (DEBUG>0):
    print ( "PROCESS_RNA_SEQ:        INFO: will look recursively under  {:}'{:}'{:} for files that match this pattern: {:}{:}{:}".format( BB, data_dir, RESET, BB, rna_file_suffix, RESET ),  flush=True ) 

  # STEP 1: READ ENSEMBL FROM pmcc_reference_file; REMOVE BLANKS; CONVERT TO NUMPY VECTOR
  pmcc_genes_of_interest = pd.read_csv(pmcc_genes_reference_file, sep='\t', na_filter=False )

  if DEBUG>99:
    print ( f"PROCESS_RNA_SEQ: pmcc_genes_of_interest as pandas object = \n\033[35m{pmcc_genes_of_interest}\033[m" )

  np_pmcc_reference   = pmcc_genes_of_interest.to_numpy()

  if DEBUG>99:
    print ( f"PROCESS_RNA_SEQ: pmcc_genes_of_interest as numpy array = \n\033[35m{np_pmcc_reference}\033[m" )
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference.shape           = \033[35m{np_pmcc_reference.shape}\033[m" )

  np_pmcc_reference_as_vector   = np.concatenate(np_pmcc_reference)

  if DEBUG>99:
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector = \n\033[35m{np_pmcc_reference_as_vector}\033[m" )
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector.shape = \033[35m{np_pmcc_reference_as_vector.shape}\033[m" )

  np_pmcc_reference_as_vector = [i for i in np_pmcc_reference_as_vector if "ENSG" in i ]

  if DEBUG>99:
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector with empty strings removed       = \n\033[35m{np_pmcc_reference_as_vector}\033[m" )
    print ( f"PROCESS_RNA_SEQ: len(np_pmcc_reference_as_vector with empty strings removed) = \033[35m{len(np_pmcc_reference_as_vector)}\033[m" )


  # STEP 2: OPEN RNA "FPKM_UQ" RESULTS FILE; EXTRACT ROWS WHICH CORRESPOND TO PMCC CANCER GENES OF INTEREST, SAVE AS CSV FILE WITH SAME NAME AS ORIGINAL PLUS 'REDUCED' SUFFIX
  
  last_table_shape=np.array([0,0])
  
  walker = os.walk(data_dir)
  for root, __, files in walker:
    
    for f in files:
      
      current_fqn = os.path.join( root, f)
      new_f       = f"{f}{rna_file_reduced_suffix}"
      new_fqn     = os.path.join( root, new_f)
        
      if (DEBUG>99):
        print ( "PROCESS_RNA_SEQ:        INFO: (current_fqn)                    \033[34m{:}\033[m".format(   current_fqn          ),  flush=True )  
  
      if fnmatch.fnmatch( f, rna_file_suffix  ):
   
        rna_results_file_found   =1
        cumulative_found_count    +=1  
        
        if (DEBUG>99): 
          print ( f"PROCESS_RNA_SEQ:        INFO: (match !)                         {BB}{current_fqn}{RESET}    cumulative_found_count = {BB}{cumulative_found_count}{RESET}",  flush=True )

        ensembl_gene_id_column = pd.read_csv(current_fqn, sep='\t', usecols=[0,1])
        np_ensemble_gene_ids   = ensembl_gene_id_column.to_numpy()

        if DEBUG>99:
          print ( f"PROCESS_RNA_SEQ: np_ensemble_gene_ids      = \033[35m{np_ensemble_gene_ids}\033[m" )
          print ( f"PROCESS_RNA_SEQ: len(np_ensemble_gene_ids) = \033[35m{len(np_ensemble_gene_ids)}\033[m" )
        
        new_table = np.array([len(np_pmcc_reference_as_vector),2])                                         # assumes we can't find more than 'np_pmcc_reference_as_vector' matches, which is valid
                
        for target in np_ensemble_gene_ids:
          if DEBUG>99:
            print ( f"PROCESS_RNA_SEQ:        INFO: main()              target = \033[35m{target}\033[m" )
          r=strip_suffix(target[0])
          if DEBUG>99:
            print ( f"PROCESS_RNA_SEQ:        INFO: r = {r}" )
            
          if r in np_pmcc_reference_as_vector:                                                             # if the row contains one of the PMCC cancer genes of interest
            new_table = np.vstack([new_table, target])                                                     # then add the row to new_table
            new_table_shape  = new_table.shape
            last_table_shape = new_table.shape
            if not new_table_shape==last_table_shape:
              print(f"\033[31m\033[1mTILER: FATAL: size of table that will become the reduced FPKM_UQ file {new_table_shape} differs from that of the last FPKM_UQ file processed {last_table_shape}" , flush=True)
              print(f"\033[31m\033[1mTILER: FATAL: this should not happen, and will cause training to crasg, so preemptively stopping now ",                                                            flush=True)
          else: 
            if DEBUG>99:
              print ( "PROCESS_RNA_SEQ:        INFO: \033[31;1mthis is not one of the PMCC genes of interest -- moving on \033[m" )

        if DEBUG>99:
          print ( f"PROCESS_RNA_SEQ: new_table        = \033[35m{new_table}\033[m" )
          print ( f"PROCESS_RNA_SEQ: new_table_shape  = \033[35m{new_table_shape}\033[m" )
          print ( f"PROCESS_RNA_SEQ: last_table_shape = \033[35m{last_table_shape}\033[m" )

        #time.sleep(1)

        try:
          pd.DataFrame(new_table).to_csv(new_fqn, index=False, header=False, index_label=False )                         # don't add the column and row labels that Pandas would otherwise add
          if DEBUG>0:
            print ( f"PROCESS_RNA_SEQ:        INFO: saving reduced file with dims \033[35m{new_table_shape}\033[m to name \033[35m{new_fqn}\033[m        cumulative found count = \033[35m{cumulative_found_count}\033[m"  )
        except Exception as e:
          print ( f"PROCESS_RNA_SEQ:        FATAL: could not save file         = \033[35m{new_table}\033[m"  )
          sys.exit(0)

#====================================================================================================================================================
def strip_suffix(s):
  
  if DEBUG>99:
    print ( f"PROCESS_RNA_SEQ:        INFO: strip_suffix()           s = { s }", flush=True )
  
  r=re.search('^ENSG[0-9]*', s)

  if r:
    found = r.group(0)
    if DEBUG>99:
      print ( f"PROCESS_RNA_SEQ:        INFO: strip_suffix () r.group(0) = { r.group(0) }", flush=True )
    return r.group(0)
  else:
    return 0

#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                     type=str, default="/home/peter/git/pipeline/dataset")
  p.add_argument('---pmcc_genes_reference_file',   type=str, default="/home/peter/git/pipeline/dataset/pmcc_genes_reference_file")
  p.add_argument('--rna_file_suffix',              type=str, default='*FPKM-UQ.txt')
  p.add_argument('--rna_file_reduced_suffix',      type=str, default='_reduced')
  p.add_argument('--rna_exp_column',               type=int, default=1)

  
  args, _ = p.parse_known_args()

  main(args)
      
