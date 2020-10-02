"""=============================================================================
Code to set up the master mapping file from TCGA project file applicable to the cancer type

For example, the project file for TCGA STAD is "nationwidechildrens.org_clinical_patient_stad.txt"

NOTES:
=====

1   The TCGA project file must be MANUALLY edited as follows prior to running 'create_master_mapping file'. Otherwise nothing mentioned here will work.
      a) delete one of the two header rows
      b) insert a new column with heading 'type_n'. This column will hold the class labels, which the user will manually enter
      c) insert the class for each case (based on the descriptions provided in other columns, e.g. "histologic diagnosis", "tumour_grade" and "prospective_collection" in the case of STAD)
         - classes are numbers, starting at zero, and without gaps (e.g. 0,3,5,6 is no good)
    It's not possible to generate the class labels automatically, because the text descriptions tend to be at least a little ambiguous

2   This module (create_master_mapping_file.py) will use the manually edited file described above as it's input.  It will not work unless it (i) exists and (ii) has been edited exactly as per 1. above

3   This module (create_master_mapping_file.py) will do the following:
      a) delete any existing custom mapping files which may exist in the applicable master dataset directory (e.g. "stad_global"), since these would otherwise be confusing orhpans
      b) convert all text in the 'mapping_file_master' to lower case
      c) add two new columns to the immediate right of the "type_n" column, as follows: "have_wsi", "have_rna", with the following meanings
           "have_wsi" = the case exists in the master dataset directory (e.g. "stad_global") and it contains at least one Whole Slide Image file
           "have_rna" = the case exists in the master dataset directory (e.g. "stad_global") and it contains at least one rna_seq           file
      d) scan the master dataset directory (i.e. "xxxx_global") and populate each of these two columns with a number specifying the number of WSI/rna-seq files that currently exist in the master dataset directory
      e) delete any cases (subdirectories) which may exist in the master dataset directory but which have '0' in both the "have_wsi" column and the "have_rna"
            such cases  are useless because we don't have class information for them. In theory there shouldn't be any, but in practice TCGA does contain at least some sample data that is not listed in the project spreadsheet, and it's easier to just delete them that to cater for them in downstream code
      f) output a file called 'xxxx_master_mapping_file' to the applicable master dataset directory, where xxxx is the TCGA abbreviation for the applicable cancer (e.g. 'stad_master_mapping_file')
      
4   The output file (e.g. 'stad_master_mapping_file') becomes the default mapping file to be used for an experiment job

5   However a second module - "customise_mapping_file.py" -  can optionally be used to generate a custom mapping file which may alternatively be used for an experiment job

6   The following kinds of customisations may be used either alone or in combination with "customise_mapping_file.py" to define a custom mapping file which:
      a)  removes classes which exist in very small numbers (if requested, this is done first)
      b)  define dataset comprising: 
            (i)   ALL or a specified number of just image files OR just rna_seq files OR just matched image + rna_seq files AND
            (ii)  optionally specified that it must be balanced (as defined by applicable user parameters)

7   "customise_mapping_file.py" will be a new file in  master dataset directory with a readable name indicating the nature of the customisation, as follows:
       mapping_file_custom_stad_[not_]balanced_<[image_nnn] [rna_nnn] [matched_nnn]"" interpretable as per the following examples:
       
          mapping_file_custom_stad_not_balanced_image_all       --- includes every case which has an image ; no attempt to balance classes
          mapping_file_custom_stad_balanced_rna_all             --- includes every rna file, consistent with the classes being balanced
          mapping_file_custom_stad_not_balanced_matched_all     --- includes every case which has matched image and rna_seq data 
          mapping_file_custom_stad_balanced_image_100           --- includes 100 image cases, consistent with the classes being balanced   << if theren't aren't 100, it will give a warning, use max available and name the file accordingly                    

8   note the following:
      a) neither 'create_master_mapping_file.py' nor 'customise_mapping_file.py' will change the contents of the master dataset directory (i.e. "xxxx_global") other than to delete directories that don't exist in the applicable project file
      b) downstream code should use the contents of the applicable mapping file to generate a pytorch dataset which corresponds to the mapping_file
      c) downstream code should not
          (i)   delete cases (subdirectories) from the master dataset directory
          (ii)  delete cases (subdirectories) from the working dataset directory (otherwise a new copy would have to be made for every experiment, and the copy is very time-consuming)

9  customise mapping files are used by 'generate()' in the following fashion:
      1) each time generate() traverses a new case (subdirectory), it checks to see if the case is listed in the currently selected custom mapping file
          a) if it is, it uses the files in the directory, in accordance with the 'INPUT_MODE' flag ( 'image', 'rna', 'image_rna')
          b) if it is not it skips the directory
      2) it accomplishes this by asking the a helper function "check_mapping_file( < image | rna | image_rna > ) which returns either 'True' (use it) 'False' (skip it)
    this somewhat convoluted method is used is to avoid having to re-generate the working dataset (a time consuming process) each time a different custom mapping file is selected by the user

10  user notes:
      a) if a custom file is to be used, it must (i) exist in the applicable master dataset directory (e.g. "stad_global") and (ii) be specified at MAPPING_FILE_NAME in variables.sh (e.g. mapping_file_custom_stad_not_balanced_image_all)
      b) if MAPPING_FILE_NAME is not specified, the applicable master mapping file will be used (e.g. stad_master_mapping_file). Again:
           (i)   it must exist (and it will only exist if 'create_master_mapping_file.py' has been run
           (ii)  it must be in the applicable master dataset directory (e.g. "stad_global")  ('create_master_mapping_file.py' will take care of this)
      c) if a custom mapping file is specified and there is no master mapping file, the job will still run, but this is bad practice as the custom mapping file will have an unknown 
      
      
============================================================================="""

import os
import sys
import math
import time
import pprint
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

pd.set_option('max_colwidth', 50)

#===========================================

np.set_printoptions(edgeitems=500)
np.set_printoptions(linewidth=400)

pd.set_option('display.max_rows',     50 )
pd.set_option('display.max_columns',  13 )
pd.set_option('display.width',       300 )
pd.set_option('display.max_colwidth', 99 )  

# ------------------------------------------------------------------------------

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

FAIL    = 0
SUCCESS = 1

DEBUG   = 1


# ------------------------------------------------------------------------------
def main(args):

  now = time.localtime(time.time())
  print(time.strftime("\nCREATE_MASTER:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time()

  base_dir            = args.base_dir
  data_dir            = args.data_dir
  dataset             = args.dataset
  data_source         = args.data_source
  global_data         = args.global_data
  mapping_file        = args.mapping_file  
  mapping_file_name   = args.mapping_file_name
  case_column         = args.case_column
  class_column        = args.class_column

  #n_classes=len(class_names)


  # Global settings --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

  np.set_printoptions(formatter={'float': lambda x: "{:>7.3f}".format(x)})    


    
 #pd.set_option( 'display.max_columns',    25 )
 #pd.set_option( 'display.max_categories', 24 )
 #pd.set_option( 'precision',               1 )
  pd.set_option( 'display.min_rows',    8     )
  pd.set_option( 'display.float_format', lambda x: '%6.2f' % x)    
  np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})


  if (DEBUG>0):    
    print ( f"CREATE_MASTER:     INFO: about to open:   {MAGENTA}{mapping_file}{RESET}")

  try:
    df_map = pd.read_csv( mapping_file, sep=',' )
  except Exception as e:
    print ( f"{RED}CREATE_MASTER:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  explanation: there is no mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the dataset working copy ({MAGENTA}{data_dir}{RESET}{RED}){RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: ensure there's a valid mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the {MAGENTA}{dataset}{RESET}{RED} source dataset directory ({MAGENTA}{global_data}{RESET}{RED}){RESET}" )                                   
    print ( f"{RED}CREATE_MASTER:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)  

  #gene_names_table=df_map.iloc[:,1]
  if DEBUG>0:
    print ( f"CREATE_MASTER:       INFO:      pandas description of df_map: \n{CYAN}{df_map.describe}{RESET}", flush=True )  
  if DEBUG>0:
    print ( f"CREATE_MASTER:       INFO:      df_map.shape = {CYAN}{ df_map.shape}{RESET}", flush=True )  
  if DEBUG>99:
    print ( f"CREATE_MASTER:       INFO:      start of df_map: \n{CYAN}{df_map.iloc[:,1]}{RESET}", flush=True )
  if DEBUG>99:
    print(tabulate(df_map, tablefmt='psql'))



  print( f"\n\nCREATE_MASTER:     INFO: {MIKADO}finished{RESET}" )
  hours   = round((time.time() - start_time) / 3600, 1  )
  minutes = round((time.time() - start_time) / 60,   1  )
  seconds = round((time.time() - start_time), 0  )
  #pprint.log_section('Job complete in {:} mins'.format( minutes ) )

  print(f'CREATE_MASTER:     INFO: took {MIKADO}{minutes}{RESET} mins ({MIKADO}{seconds:.1f}{RESET} secs)')
  


  
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--base_dir',                                                      type=str     )
    p.add_argument('--data_dir',                                                      type=str     )
    p.add_argument('--dataset',                                                       type=str     )
    p.add_argument('--data_source',                                                   type=str     )
    p.add_argument('--global_data',                                                   type=str     )
    p.add_argument('--mapping_file',                                                  type=str     )
    p.add_argument('--mapping_file_name',                                             type=str     )
    p.add_argument('--case_column',           type=str, default="bcr_patient_uuid"                 )
    p.add_argument('--class_column',          type=str, default="type_n"                           )

    args, _ = p.parse_known_args()


    main(args)
