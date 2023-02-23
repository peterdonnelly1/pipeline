"""
If user has opted to filter genes:

  open each rna results file remove all rows which contain genes (ENSEMBL gene IDs) that do not correspond to TARGET's cancer genes of interest; save as file with the same name but with 'reduced' suffix

     cancer genes of interest must be contained within a file called 'target_genes_of_interest' in working_data/, and must also ...
        
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

from constants  import *

DEBUG   = 1

a = random.choice( range(200,255) )
b = random.choice( range(50, 225) )
c = random.choice( range(50, 225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=275)
np.set_printoptions(threshold=10000)


#====================================================================================================================================================
def filter_genes( args ):
  
  data_dir                    = args.data_dir
  target_genes_reference_file = args.target_genes_reference_file
  random_genes_count          = args.random_genes_count
  tcga_rna_seq_file_suffix    = args.tcga_rna_seq_file_suffix
  tcga_rna_seq_metric         = args.tcga_rna_seq_metric
  tcga_rna_seq_start_row      = args.tcga_rna_seq_start_row
  rna_file_reduced_suffix     = args.rna_file_reduced_suffix
  use_unfiltered_data         = args.use_unfiltered_data
  low_expression_threshold    = args.low_expression_threshold
  skip_rna_preprocessing      = args.skip_rna_preprocessing


  if  skip_rna_preprocessing == True:
    print( f"{ORANGE}FILTER_GENES:   INFO: '{CYAN}skip_rna_preprocessing{RESET}{ORANGE}' flag = {MIKADO}{skip_rna_preprocessing}{RESET}{ORANGE}. No gene filtering will be performed, and '{MAGENTA}_reduced{RESET}{ORANGE}' files will NOT be generated. {RESET}" )
    print( f"{ORANGE}FILTER_GENES:   INFO: This may be intentional on your part: the required files may alreay exist, and you may be using this flag to avoid repeatedly generating the same (gene filtered) files again. {RESET}" )
    return

  if  use_unfiltered_data == True:
    print( f"{ORANGE}FILTER_GENES:   INFO: '{CYAN}use_unfiltered_data{RESET}{ORANGE}' flag = {MIKADO}{use_unfiltered_data}{RESET}{ORANGE}. No gene filtering will be performed, and '{MAGENTA}_reduced{RESET}{ORANGE}' files will NOT be generated. {RESET}" )
    return
  else:
    print( f"{CHARTREUSE}{BOLD}FILTER_GENES:   INFO: '{CYAN}{BOLD}use_unfiltered_data{RESET}{CHARTREUSE}{BOLD}' flag = {MIKADO}{use_unfiltered_data}{RESET}{CHARTREUSE}{BOLD}, so filtering will be performed{RESET}{CHARTREUSE}{BOLD} and '{MAGENTA}{BOLD}_reduced{RESET}{CHARTREUSE}{BOLD}' files will be generated and saved to the working dataset {RESET}" )    


  if (DEBUG>99):
    print ( f"FILTER_GENES:   INFO: target_genes_reference_file    = {MAGENTA}{os.path.basename(target_genes_reference_file)}{RESET}",  flush=True )





  # STEP 0: if user has requested that a random selection of genes be used (RANDOM_GENES_COUNT>0), we CONSTRUCT a reference file comprising the user stipulated number of randomly selected of genes

  if os.path.basename( target_genes_reference_file ) == 'randomly_selected_genes':

    if DEBUG>9999:
      print ( f"\n{BOLD}{UNDER}FILTER_GENES:           INFO: STEP 0: User has requested that {MAGENTA}{random_genes_count}{RESET}{BOLD}{UNDER} randomly selected genes be used{RESET}\n" )
    
    cases_processed_count = 0

    # Any old RNA-Seq gene vector file will do, as they all list the same set of genes
    
    for root, __, files in os.walk(args.data_dir):  
    
      for f in files:
          
        rna_seq_file_unfiltered_fqn = os.path.join( root, f)
    
        if fnmatch.fnmatch( f, args.tcga_rna_seq_file_suffix ):
     
          cases_processed_count  +=1 
          
          if (DEBUG>4444):
            print ( f"FILTER_GENES:           INFO: will extract the random genes that will be used from this file  {MAGENTA}{rna_seq_file_unfiltered_fqn}{RESET}",  flush=True )  

          rna_seq_file_unfiltered = pd.read_csv( rna_seq_file_unfiltered_fqn, sep='\t', usecols=[8], header=None )            
  
          if DEBUG>0:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_unfiltered.shape         = {MIKADO}{rna_seq_file_unfiltered.shape}{RESET}" )

          if DEBUG>12000:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_unfiltered contents      = \n{MIKADO}{rna_seq_file_unfiltered.head(10)}{RESET}" )


          rna_seq_file_trimmed = pd.DataFrame([  r for r in rna_seq_file_unfiltered.iloc[:,0].str.slice(0, 15 )  ]   )              # trim version numbers off the ENSG column (take the first 15 characters, leave the portion after the dot behind)

          if DEBUG>444:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_trimmed.shape            = {MIKADO}{rna_seq_file_trimmed.shape}{RESET}" )

          if DEBUG>12000:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_trimmed contents         = \n{MIKADO}{rna_seq_file_trimmed.head(10)}{RESET}" )
            

          # Extract the required number of genes from rna_seq_file_unfiltered_fqn and save as the same file name (which will be target_genes_reference_file = 'randomly_selected_genes')
          
          rna_seq_file_filtered = rna_seq_file_trimmed.sample( n=random_genes_count, replace=False)          
    
          if DEBUG>444:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_filtered.shape           = {MIKADO}{rna_seq_file_filtered.shape}{RESET}" )

          if DEBUG>12000:
            print ( f"FILTER_GENES:           INFO: rna_seq_file_filtered                 = \n{AMETHYST}{rna_seq_file_filtered}{RESET}" )
   
    
          try:
            rna_seq_file_filtered.to_csv( args.target_genes_reference_file, index=False, header=False, index_label=False  )           # don't add the column and row labels that Pandas would otherwise add
            if DEBUG>1:
              print ( f"FILTER_GENES:           INFO: saving case with dims {MIKADO}{rna_seq_file_filtered.shape}{RESET} to name {MAGENTA}{args.target_genes_reference_file}{RESET}"  )
          except Exception as e:
            print ( f"{RED}FILTER_GENES:           FATAL: could not save file            = {CYAN}{rna_seq_file_filtered}{RESET}"  )
            print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
            print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
            print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
            print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
            print ( f"{RED}FILTER_GENES:                FATAL:  The reported error was: {CYAN}{e}{RESET}" )
            print( f"{RED}FILTER_GENES:                 FATAL:  ... halting now\n\n\n\n\n\n {RESET}" )
            sys.exit(0)
  
        if cases_processed_count > 0:                                   # 1 is all we need
          break 

      if cases_processed_count > 0:                                     # 1 is all we need
        break 

  
  result = reduce_genes( args, target_genes_reference_file )

  if result==FAIL:
    print( f"{RED}FILTER_GENES:           FATAL:   reduce_genes() returned 'FAIL' ... halting now {RESET}" )
    sys.exit(0)

  if DEBUG>9999:
    print ( f"\n{BOLD}{UNDER}FILTER_GENES:           INFO: STEP 1: READ GENE NAMES FROM {CYAN}TARGET_REFERENCE_FILE{RESET}{BOLD}{UNDER} INTO A PANDAS OBJECT; REMOVE BLANKS; CONVERT TO NUMPY VECTOR{RESET}\n" )





def reduce_genes( args, target_genes_reference_file ):
  
  # STEP 1: READ gene names FROM target_reference_file into a Pandas object; REMOVE BLANKS; CONVERT TO NUMPY VECTOR

  if DEBUG>9999:
    print ( f"\n{BOLD}{UNDER}FILTER_GENES:           INFO: STEP 1: READ GENE NAMES FROM {CYAN}TARGET_REFERENCE_FILE{RESET}{BOLD}{UNDER} INTO A PANDAS OBJECT; REMOVE BLANKS; CONVERT TO NUMPY VECTOR{RESET}\n" )


  if (DEBUG>5):
    print ( f"{ORANGE}FILTER_GENES:           INFO: will look recursively under {MAGENTA}'{args.data_dir}'{ORANGE} for files with the TCGA RNA-Seq suffix: {BB}{args.tcga_rna_seq_file_suffix}{RESET}",  flush=True ) 

  if (DEBUG>99):
    print ( f"FILTER_GENES:           INFO: target_genes_reference_file     = {MAGENTA}{args.target_genes_reference_file}{RESET}",  flush=True )
    print ( f"FILTER_GENES:           INFO: args.data_dir                   = {MAGENTA}{args.data_dir}{RESET}",                     flush=True )
    print ( f"FILTER_GENES:           INFO: args.tcga_rna_seq_file_suffix   = {MAGENTA}{args.tcga_rna_seq_file_suffix}{RESET}",     flush=True )
    print ( f"FILTER_GENES:           INFO: args.tcga_rna_seq_metric = {MIKADO}{args.tcga_rna_seq_metric}{RESET}",    flush=True )
    print ( f"FILTER_GENES:           INFO: args.tcga_rna_seq_start_row     = {MIKADO}{args.tcga_rna_seq_start_row}{RESET}",        flush=True )

  
  try:
    target_genes_of_interest = pd.read_csv( target_genes_reference_file, sep='\t', na_filter=False, header=None )
  except Exception as e:
    print ( f"{RED}FILTER_GENES:                FATAL: {CYAN}pd.read_csv{RESET}{RED} failed when trying to open {CYAN}{args.target_genes_reference_file}{RESET}"  )
    print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
    print ( f"{RED}FILTER_GENES:                FATAL:  The reported error was: {CYAN}{e}{RESET}" )
    print( f"{RED}FILTER_GENES:                 FATAL:  ... halting now\n\n\n\n\n {RESET}" )
    time.sleep(10)
    sys.exit(0)


  if DEBUG>12000:
    print ( f"FILTER_GENES:           INFO: target_genes_of_interest as pandas object = \n\033[35m{target_genes_of_interest}\033[m" )

  reference_genes                  = target_genes_of_interest.to_numpy()

  if DEBUG>12000:
    print ( f"FILTER_GENES:           INFO: target_genes_of_interest as numpy array = \n\033[35m{reference_genes}\033[m" )
  if DEBUG>999:
    print ( f"FILTER_GENES:           INFO: reference_genes.shape                = {ARYLIDE}{reference_genes.shape}{RESET}" )

  genes_of_interest_concatenated   = np.concatenate(reference_genes)

  if DEBUG>12000:
    print ( f"FILTER_GENES:           INFO: genes_of_interest_concatenated       = \n\033[35m{genes_of_interest_concatenated}\033[m" )
  if DEBUG>999:
    print ( f"FILTER_GENES:           INFO: genes_of_interest_concatenated.shape = {ARYLIDE}{genes_of_interest_concatenated.shape}{RESET}" )

  genes_of_interest_concatenated = [i for i in genes_of_interest_concatenated if "ENSG" in i ]

  print( f"FILTER_GENES:           INFO: the user provided file which lists ENSG gene names to keep; namely: {CYAN}{BOLD}TARGET_GENES_REFERENCE_FILE{RESET}={MAGENTA}{BOLD}{os.path.basename(target_genes_reference_file)}{RESET}, contains {MIKADO}{len(genes_of_interest_concatenated):,}{RESET} genes. (location: '{MAGENTA}{target_genes_reference_file}{RESET}'){RESET}" )    


  if DEBUG>999:
    print ( f"FILTER_GENES:           INFO: genes_of_interest_concatenated with empty strings removed        = \n{MIKADO}{genes_of_interest_concatenated}{RESET}" )
  if DEBUG>999:
    print ( f"FILTER_GENES:           INFO: {CYAN}len(genes_of_interest_concatenated){RESET} (with empty strings removed) = {MIKADO}{len(genes_of_interest_concatenated)}{RESET}" )




  # STEP 2: EXTRACT ROWS CORRESPONDING TO TARGET GENES OF INTEREST FROM RNA_SEQ FILE, SAVE AS (TSV) FILE WITH SAME NAME AS ORIGINAL PLUS 'REDUCED' SUFFIX

  if DEBUG>9999:
    print ( f"\n{BOLD}{UNDER}FILTER_GENES:           INFO: STEP 2: OPEN RNA '{CYAN}{tcga_rna_seq_file_suffix}{RESET}{BOLD}{UNDER} RESULTS FILE; EXTRACT ROWS WHICH CORRESPOND TO TARGET CANCER GENES OF INTEREST, SAVE AS (TSV) FILE WITH SAME NAME AS ORIGINAL PLUS '{CYAN}_REDUCED{RESET}{BOLD}{UNDER}' SUFFIX{RESET}\n" )

  pmcc_reference    = pd.DataFrame( genes_of_interest_concatenated )
  
  if DEBUG>999:
    print ( f"FILTER_GENES:           INFO: pmcc_reference.shape = {ARYLIDE}{pmcc_reference.shape}{RESET}" )


  cases_processed_count = 0
  
  for root, __, files in os.walk(args.data_dir):
    
    for f in files:
      
      rna_seq_file_unfiltered_fqn = os.path.join( root, f)
      rna_seq_file_filtered_fqn   = os.path.join( root, f"{f}{args.rna_file_reduced_suffix}")
        
      if (DEBUG>11000):
        print ( f"FILTER_GENES:           INFO: (rna_seq_file_unfiltered_fqn)                    {MAGENTA}{rna_seq_file_unfiltered_fqn}{RESET}",  flush=True )  
  
      if fnmatch.fnmatch( f, args.tcga_rna_seq_file_suffix ):                                                       # case
   
        cases_processed_count  +=1  
        
        if (DEBUG>4444):
          print ( f"FILTER_GENES:           INFO: (match !)                              {CARRIBEAN_GREEN}{rna_seq_file_unfiltered_fqn}{RESET}    \r\033[210Ccases_processed_count = {MIKADO}{cases_processed_count}{RESET}",  flush=True )

        rna_seq_file_unfiltered = pd.read_csv( rna_seq_file_unfiltered_fqn, sep='\t', usecols=[0,args.tcga_rna_seq_metric], header=None )
        
        ensg  = pd.DataFrame([  r for r in rna_seq_file_unfiltered.iloc[:,0                         ].str.slice(0, 15 )  ]   )        # trim version numbers off the ENSG column (take the first 15 characters, leave the portion after the dot behind)
        value = pd.DataFrame([  r for r in rna_seq_file_unfiltered.iloc[:,args.tcga_rna_seq_metric]                    ]   )
        
        rna_seq_file_trimmed = pd.concat(  [ ensg , value ], axis=1 )
        
        if DEBUG>444:
          print ( f"FILTER_GENES:           INFO: rna_seq_file_unfiltered.shape         = {MIKADO}{rna_seq_file_unfiltered.shape}{RESET}" )
          print ( f"FILTER_GENES:           INFO: ensg  (Ensembl name)   shape          = {MIKADO}{ensg .shape}{RESET}" )                
          print ( f"FILTER_GENES:           INFO: value (FPKM UQ value)  shape          = {MIKADO}{value.shape}{RESET}" ) 
          print ( f"FILTER_GENES:           INFO: rna_seq_file_trimmed    shape         = {AMETHYST}{rna_seq_file_trimmed.shape}{RESET}" ) 

        if DEBUG>12000:
          print ( f"FILTER_GENES:           INFO: rna_seq_file_unfiltered contents      = \n{MIKADO}{rna_seq_file_unfiltered.head(10)}{RESET}" )  
          print ( f"FILTER_GENES:           INFO: ensg  (Ensembl name)   contents       = {MIKADO}{ensg }{RESET}" )                
          print ( f"FILTER_GENES:           INFO: value (FPKM UQ value)  contents       = {MIKADO}{value}{RESET}" )                
          print ( f"FILTER_GENES:           INFO: rna_seq_file_trimmed                  = \n{AMETHYST}{rna_seq_file_trimmed}{RESET}" )    

        rna_seq_file_filtered = rna_seq_file_trimmed[ (rna_seq_file_trimmed.iloc[:,0]).isin(pmcc_reference.iloc[:,0]) ]     # <<<<<<<<<<<<<<<<<<<<<< DOES THE WORK

        if DEBUG>444:
          print ( f"FILTER_GENES:           INFO: rna_seq_file_filtered.shape           = {MIKADO}{rna_seq_file_filtered.shape}{RESET}" )
          
        if DEBUG>12000:
          print ( f"FILTER_GENES:           INFO: rna_seq_file_filtered                 = \n{AMETHYST}{rna_seq_file_filtered}{RESET}"   )          
  
        try:
          rna_seq_file_filtered.to_csv( rna_seq_file_filtered_fqn, index=False, header=False, index_label=False )     # suppress the column and row labels that Pandas would otherwise add
          if DEBUG>1:
            print ( f"FILTER_GENES:           INFO: saving case with dims {MIKADO}{rna_seq_file_filtered.shape}{RESET} to name  {MAGENTA}{rna_seq_file_filtered_fqn}{RESET}        \r\033[210Ccases_processed_count = {MIKADO}{cases_processed_count}{RESET}"  )
        except Exception as e:
          print ( f"{RED}FILTER_GENES:           FATAL: could not save file            = {CYAN}{rna_seq_file_filtered}{RESET}"  )
          print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}FILTER_GENES:                FATAL:     ^^^^ NOTICE THE ABOVE FATAL ERROR MESSAGE{RESET}"  )
          print ( f"{RED}FILTER_GENES:                FATAL:  The reported error was: {CYAN}{e}{RESET}" )
          print( f"{RED}FILTER_GENES:                 FATAL:  ... halting now\n\n\n\n\n\n {RESET}" )
          sys.exit(0)


      if (DEBUG>0):
        if cases_processed_count % 25 == 0:
          print ( f"FILTER_GENES:           INFO: {MIKADO}{cases_processed_count}{RESET} cases (RNA-Seq files) processed and filtered versions saved. Filtering takes a minute or two per thousand cases depending on how many/fast CPUs",  flush=True )
          print ( "\033[2A",  flush=True )

  if (DEBUG>0):                                                                                          # this will show the final count
    print ( f"{CLEAR_LINE}FILTER_GENES:           INFO: {MIKADO}{cases_processed_count}{RESET} cases (RNA-Seq files) processed and filtered versions saved",  flush=True )
    print ( "\033[2A",  flush=True )

  if (DEBUG>0):
    print ( "\033[1B",  flush=True )

  
  # ~ if (DEBUG>0):
    # ~ print ( f"{ORANGE}FILTER_GENES:           INFO: {MIKADO}{cases_processed_count}{RESET}{ORANGE} cases found and processed{RESET}",  flush=True )  
        
  return SUCCESS

#====================================================================================================================================================
def strip_suffix(s):
  
  if DEBUG>9999:
    print ( f"FILTER_GENES:           INFO: strip_suffix()           s = { s }", flush=True )
  
  value=re.search('^ENS[A-Z][0-9]*', s)

  if value:
    found = value.group(0)
    if DEBUG>9999:
      print ( f"FILTER_GENES:           INFO: strip_suffix () value.group(0) = { value.group(0) }", flush=True )
    return value.group(0)
  else:
    return 0
