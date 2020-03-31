"""
open each rna results file within the patches directory; ...

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
  rna_file_suffix             = args.rna_file_suffix
  rna_ensembl_gene_id_column  = args.rna_ensembl_gene_id_column
  rna_exp_column              = args.rna_exp_column
  
  if (DEBUG>0):
    print ( "PROCESS_RNA_SEQ:        INFO: args.data_dir               = {:}{:}{:}".format( BB, data_dir,           RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_file_suffix        = {:}{:}{:}".format( BB, rna_file_suffix ,   RESET ),  flush=True )
    print ( "PROCESS_RNA_SEQ:        INFO: args.rna_exp_column         = {:}{:}{:}".format( BB, rna_exp_column,     RESET ),  flush=True )

  if (DEBUG>0):
    print ( "PROCESS_RNA_SEQ:        INFO: will look recursively under  {:}'{:}'{:} for files that match this pattern: {:}{:}{:}".format( BB, data_dir, RESET, BB, rna_file_suffix, RESET ),  flush=True ) 

  pmcc_reference_file = "sarc_global/bioDBnet_db2db_200330053818_2032226390_human.xls"
  pmcc_genes_of_interest = pd.read_csv(pmcc_reference_file, sep='\t', na_filter=False )

  if DEBUG>0:
    print ( f"PROCESS_RNA_SEQ: pmcc_genes_of_interest as pandas object = \n\033[35m{pmcc_genes_of_interest}\033[m" )

  np_pmcc_reference   = pmcc_genes_of_interest.to_numpy()

  if DEBUG>0:
    print ( f"PROCESS_RNA_SEQ: pmcc_genes_of_interest as numpy array = \n\033[35m{np_pmcc_reference}\033[m" )
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference.shape           = \033[35m{np_pmcc_reference.shape}\033[m" )

  np_pmcc_reference_as_vector   = np.concatenate(np_pmcc_reference)

  if DEBUG>0:
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector = \n\033[35m{np_pmcc_reference_as_vector}\033[m" )
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector.shape = \033[35m{np_pmcc_reference_as_vector.shape}\033[m" )

  np_pmcc_reference_as_vector = [i for i in np_pmcc_reference_as_vector if "ENSG" in i ]

  if DEBUG>0:
    print ( f"PROCESS_RNA_SEQ: np_pmcc_reference_as_vector with empty strings removed       = \n\033[35m{np_pmcc_reference_as_vector}\033[m" )
    print ( f"PROCESS_RNA_SEQ: len(np_pmcc_reference_as_vector with empty strings removed) = \033[35m{len(np_pmcc_reference_as_vector)}\033[m" )


  
  walker = os.walk(data_dir)
  for root, __, files in walker:
    for f in files:
      current_file    = os.path.join( root, f)
  
      if (DEBUG>99):
        print ( "PROCESS_RNA_SEQ:        INFO: (current_file)                    \033[34m{:}\033[m".format(   current_file          ),  flush=True )  
  
      if fnmatch.fnmatch( f, rna_file_suffix  ):
   
        rna_results_file_found   =1
        cumulative_found_count    +=1  
        
        if (DEBUG>99): 
          print ( f"PROCESS_RNA_SEQ:        INFO: (match !)                         {BB}{current_file}{RESET}    cumulative_found_count = {BB}{cumulative_found_count}{RESET}",  flush=True )

        ensembl_gene_id_column = pd.read_csv(current_file, sep='\t', usecols=[0])
        np_ensemble_gene_ids   = ensembl_gene_id_column.to_numpy()

        if DEBUG>0:
          print ( f"PROCESS_RNA_SEQ: np_ensemble_gene_ids      = \033[35m{np_ensemble_gene_ids}\033[m" )
          print ( f"PROCESS_RNA_SEQ: len(np_ensemble_gene_ids) = \033[35m{len(np_ensemble_gene_ids)}\033[m" )
        
        new=[]
                
        for target in np_ensemble_gene_ids:
          if DEBUG>99:
            print ( f"PROCESS_RNA_SEQ:        INFO: main()              target = \033[35m{target}\033[m" )
          r=strip_suffix(target[0])
          if DEBUG>99:
            print ( f"PROCESS_RNA_SEQ:        INFO: r = {r}" )
          if r in np_pmcc_reference_as_vector:
            print ( "PROCESS_RNA_SEQ:        INFO: \033[32;1mwhy yes it is\033[m" )
            new.append(r)
          else:
            if DEBUG>99:
              print ( "PROCESS_RNA_SEQ:        INFO: \033[31;1mafraid not !\033[m" )

        if DEBUG>0:
          print ( f"PROCESS_RNA_SEQ: new      = \033[35m{new}\033[m" )
          print ( f"PROCESS_RNA_SEQ: len(new) = \033[35m{len(new)}\033[m" )

        time.sleep(2)

        #new2 = [i for i in np_ensemble_gene_ids if i in np_pmcc_reference_as_vector]

        #if DEBUG>0:
         # print ( f"PROCESS_RNA_SEQ: new2     = \n\033[35m{new2}\033[m" )
          #print ( f"PROCESS_RNA_SEQ: len(new2) = \n\033[35m{len(new2)}\033[m" )

        
        #np.save(rna_npy_file, rna)

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

  p.add_argument('--data_dir',                     type=str, default="/home/peter/git/pipeline/dataseff")
  p.add_argument('--rna_file_suffix',              type=str, default="*FPKM-UQ.txt")
  p.add_argument('--rna_ensembl_gene_id_column',   type=str, default=0)
  p.add_argument('--rna_exp_column',               type=str, default=1)

  
  args, _ = p.parse_known_args()

  main(args)
      
