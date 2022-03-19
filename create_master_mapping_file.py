"""=============================================================================
Code to set up the master mapping file from TCGA project file containing clinical data which is applicable to the cancer type

For example,the project file for TCGA STAD is "nationwidechildrens.org_clinical_patient_stad.txt"

NOTES:
=====

1   The TCGA project file must be MANUALLY edited as follows prior to running 'create_master_mapping file'. Otherwise nothing mentioned here will work.
      a) insert a new column with heading 'type_n'. This column will hold the class labels, which the user will manually enter
      b) insert the class for each case (based on the descriptions provided in other columns, e.g. "histologic diagnosis" in the case of STAD)
         - classes must be numbers, starting at zero, and without gaps (e.g. 0,3,5,6 is no good)
    It's not possible to generate the class labels automatically, because the text descriptions tend to be at least a little ambiguous, and often very ambiguous and overlapping

2   This module (create_master_mapping_file.py) will use the manually edited file described above as it's input.  It will not work unless it (i) exists and (ii) has been edited exactly as per 1. above

3   This module (create_master_mapping_file.py) will do the following:
      a) delete any existing custom mapping files which may exist in the applicable global data directory (e.g. "stad_global"), since these would otherwise be confusing orhpans
      b) convert all text in the 'mapping_file_master' to lower case
      c) add two new columns to the immediate right of the "type_n" column, as follows: "have_wsi", "have_rna", with the following meanings
           "have_wsi" = the case exists in the master dataset directory (e.g. "stad_global") and it contains at least one Whole Slide Image file
           "have_rna" = the case exists in the master dataset directory (e.g. "stad_global") and it contains at least one rna_seq           file
      d) scan the master dataset directory (i.e. "xxxx_global") and populate each of these two columns with a number specifying the number of WSI/rna-seq files that currently exist in the master dataset directory
      e) delete any cases (subdirectories) which may exist in the master dataset directory but which have '0' in both the "have_wsi" column and the "have_rna"
            such cases  are useless because we don't have class information for them. In theory there shouldn't be any, but in practice TCGA does contain at least some sample data that is not listed in the project spreadsheet, and it's easier to just delete them that to cater for them in downstream code
      f) output a file called 'xxxx_master_mapping_file' to the applicable master dataset directory, where xxxx is the TCGA abbreviation for the applicable cancer (e.g. 'stad_master_mapping_file')
      
4   The output file (e.g. 'stad_master_mapping_file') is the default mapping file used by the expertiment platform for classification experiments. It has no file extension.



  vvvvvvvvvvvvvvvvvvvvvvvvv  NOT IMPLEMENTED FROM HERE ON vvvvvvvvvvvvvvvvvvvvvvvvvvvvv

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
import re
import sys
import glob
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

WHITE           ='\033[37;1m'
PURPLE          ='\033[35;1m'
DIM_WHITE       ='\033[37;2m'
CYAN            ='\033[36;1m'
PALE_RED        ='\033[31m'
PALE_GREEN      ='\033[32m'
AUREOLIN        ='\033[38;2;253;238;0m' 
DULL_WHITE      ='\033[38;2;140;140;140m'
MIKADO          ='\033[38;2;255;196;12m'
AZURE           ='\033[38;2;0;127;255m'
AMETHYST        ='\033[38;2;153;102;204m'
ASPARAGUS       ='\033[38;2;135;169;107m'
CHARTREUSE      ='\033[38;2;223;255;0m'
COQUELICOT      ='\033[38;2;255;56;0m'
COTTON_CANDY    ='\033[38;2;255;188;217m'
HOT_PINK        ='\033[38;2;255;105;180m'
CAMEL           ='\033[38;2;193;154;107m'
MAGENTA         ='\033[38;2;255;0;255m'
YELLOW          ='\033[38;2;255;255;0m'
DULL_YELLOW     ='\033[38;2;179;179;0m'
ARYLIDE         ='\033[38;2;233;214;107m'
BLEU            ='\033[38;2;49;140;231m'
DULL_BLUE       ='\033[38;2;0;102;204m'
RED             ='\033[38;2;255;0;0m'
PINK            ='\033[38;2;255;192;203m'
BITTER_SWEET    ='\033[38;2;254;111;94m'
DARK_RED        ='\033[38;2;120;0;0m'
ORANGE          ='\033[38;2;255;103;0m'
PALE_ORANGE     ='\033[38;2;127;63;0m'
GOLD            ='\033[38;2;255;215;0m'
GREEN           ='\033[38;2;19;136;8m'
BRIGHT_GREEN    ='\033[38;2;102;255;0m'
CARRIBEAN_GREEN ='\033[38;2;0;204;153m'
GREY_BACKGROUND ='\033[48;2;60;60;60m'


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
  print(time.strftime(f"\nCREATE_MASTER:     INFO:    {MIKADO}%Y-%m-%d %H:%M:%S %Z{RESET}", now))
  start_time = time.time()

  base_dir            = args.base_dir
  dataset             = args.dataset
  case_column         = args.case_column
  class_column        = args.class_column
  image_column        = 3
  rna_seq_column      = 4

  #n_classes=len(class_names)


  # Global settings --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

  np.set_printoptions(formatter={'float': lambda x: "{:>7.3f}".format(x)})    


    
 #pd.set_option( 'display.max_columns',    25 )
 #pd.set_option( 'display.max_categories', 24 )
 #pd.set_option( 'precision',               1 )
  pd.set_option( 'display.min_rows',    8     )
  pd.set_option( 'display.float_format', lambda x: '%6.2f' % x)    
  np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})


  cancer_class                          = dataset
  class_specific_dataset_files_location = f"{base_dir}/{dataset}"
  class_specific_global_data_location   = f"{base_dir}/{dataset}_global"
  print ( f"CREATE_MASTER:     INFO:    cancer class (from TCGA master spreadsheet as edited)  =  {CYAN}{cancer_class}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    class_specific_global_data_location                    =  {CYAN}{class_specific_global_data_location}{RESET}" )    
  print ( f"CREATE_MASTER:     INFO:    class_specific_dataset_files_location                  =  {CYAN}{class_specific_dataset_files_location}{RESET}" )   

  
  if os.path.isdir(class_specific_global_data_location)==False:
    print ( f"{RED}CREATE_MASTER:     FATAL:  the expected global data sub-directory for cancer project '{MAGENTA}{cancer_class}{RESET}{RED}', namely, '{MAGENTA}{class_specific_global_data_location}{RESET}{RED}' does not exist.{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: (i)   create a directory under '{MAGENTA}pipeline{RESET}{RED}' with name '{MAGENTA}{dataset}_global{RESET}{RED}'{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: (ii)  place a copy of the TCGA master clinical data spreadsheets file applicable to '{MAGENTA}{cancer_class}{RESET}{RED}' in this directory{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy:           The TCGA master clinical spreadsheet for '{MAGENTA}{cancer_class}{RESET}{RED}' will have a filename similar to this '{CYAN}nationwidechildrens.org_clinical_patient_{dataset}.csv{RESET}{RED}'{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy:           TCGA master clinical data spreadsheets can be found at the NIH GDC data repository: '{CYAN}https://portal.gdc.cancer.gov/repository{RESET}{RED}' (clickable link){RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy:           Local 'convenience copies' of TCGA master clinical data spreadsheets are stored in: '{CYAN}all_tcga_project_level_files/{RESET}{RED}' , hoever it is preferable to download a fresh copy from the GDC in case there have been changes{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: (iii) instructions on how to manually adjust the master spreadsheet can be found in the comments section of this ({MAGENTA}create_master_mapping_file.py{RESET}{RED}) module{RESET}" )                                        
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy:       the adjustments are mandatory. Experiments cannot work unless they are made{RESET}" )                                        
    print ( f"{RED}CREATE_MASTER:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)    

  master_spreadsheet_found=False
  for f in os.listdir( class_specific_global_data_location ):
    if f.endswith(f"_{dataset}.csv"):                                                                      # we can't be sure of the exact name, but we know it must end like this
      master_spreadsheet_found=True
      master_spreadsheet = f
      print ( f"CREATE_MASTER:     INFO:    proceeding with master spreadsheet                        '{MAGENTA}{master_spreadsheet}{RESET}'" )
      print ( f"CREATE_MASTER:     INFO:    now looking for {CYAN}{dataset}{RESET} master clinical data spreadsheet, which is assumed to be the only file in '{MAGENTA}{dataset}_global{RESET}' ending with '{MAGENTA}_{dataset}.csv{RESET}'" )
      break
      
  if master_spreadsheet_found==False:
    print ( f"{RED}CREATE_MASTER:     FATAL:  could not find the '{MAGENTA}{cancer_class}{RESET}{RED}' master spreadsheet in {MAGENTA}{class_specific_global_data_location}{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: ensure there's a valid master spreadsheet with the extension {CYAN}.csv{RESET}{RED} in {MAGENTA}{class_specific_global_data_location}{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  instructions on how to construct a master spreadsheet can be found in the comments of this ({MAGENTA}create_master_mapping_file.py{RESET}{RED}) module{RESET}" )                                        
    print ( f"{RED}CREATE_MASTER:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)       
  else:
    print ( f"CREATE_MASTER:     INFO:    have now found  {CYAN}{dataset}{RESET} master clinical data spreadsheet, which has the name '{MAGENTA}{master_spreadsheet}{RESET}'" )


  fqn = f"{class_specific_global_data_location}/{master_spreadsheet}"
  if (DEBUG>0):
    print ( f"CREATE_MASTER:     INFO:    about to open:                                            '{MAGENTA}{fqn}{RESET}'")

  try:
    df = pd.read_csv( f"{fqn}", sep=',' )
  except Exception as e:
    print ( f"{RED}CREATE_MASTER:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  explanation: there is no mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the dataset working copy ({MAGENTA}{data_dir}{RESET}{RED}){RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  remedy: ensure there's a valid mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the {MAGENTA}{dataset}{RESET}{RED} source dataset directory ({MAGENTA}{global_data}{RESET}{RED}){RESET}" )                                   
    print ( f"{RED}CREATE_MASTER:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)  

  if DEBUG>0:
    print ( f"CREATE_MASTER:     INFO:      df.shape = {CYAN}{ df.shape}{RESET}", flush=True )  
  if DEBUG>9:
    print ( f"CREATE_MASTER:     INFO:      pandas description of df: \n{CYAN}{df.describe}{RESET}", flush=True )  
  if DEBUG>99:
    print ( f"CREATE_MASTER:     INFO:      start of df: \n{CYAN}{df.iloc[:,1]}{RESET}", flush=True )
  if DEBUG>99:
    print(tabulate(df, tablefmt='psql'))

 
  df.insert(loc=3, column='image',     value='')                                                           # insert new column to hold image counts
  df.iloc[0,3]='image'
  df.iloc[1,3]='image'
  df.iloc[2:,3]=0
  df.insert(loc=4, column='rna_seq',   value='')                                                           # insert new column to hold rna_seq counts
  df.iloc[0,4]='rna_seq'
  df.iloc[1,4]='rna_seq'
  df.iloc[2:,4]=0
  df = df.fillna('').astype(str).apply(lambda x: x.str.lower())
  
  found_cases                    = 0
  found_clone_directories        = 0
  found_non_clone_directories    = 0  
  global_found_slide_file        = 0
  global_found_rna_seq_file      = 0
  global_found_file_of_interest  = 0
  global_other_files_found       = 0
  matched_cases_count            = 0

  for i in range(2, len(df)):                                                                              # for each case (row) listed in the master spreadsheet
    case =  df.iloc[i, 1]

    fqn = f"{class_specific_dataset_files_location}/{case}"
    found_cases+=1
    if DEBUG>99:
      print(fqn)

    matches = glob.glob( f"{fqn}*" )                                                                       # picks up the extra directories for the cases where there is more than one slide file. These have the extension "_<n>"
    if len(matches)>0:
      if DEBUG>9:
        print ( f"{BLEU}{matches}{RESET}"  )   

      clone_found_slide_file       = 0                                                                     # total for all clone directories. this is the value we record in the master spreadsheet
      clone_found_rna_seq_file     = 0                                                                     # total for all clone directories. this is the value we record in the master spreadsheet
      clone_found_file_of_interest = 0                                                                     # total for all clone directories. this is the value we record in the master spreadsheet

      if os.path.isdir(matches[0]):
        found_non_clone_directories+=1
          
      for j in range( 0, len(matches)) :

        found_slide_file            = 0                                                                    # total for all clone directories. this is the value we record in the master spreadsheet
        found_rna_seq_file          = 0                                                                    # total for all clone directories. this is the value we record in the master spreadsheet
        found_file_of_interest      = 0                                                                    # total for all clone directories. this is the value we record in the master spreadsheet
        if DEBUG>9:
          print ( f"{ARYLIDE}{matches[j]}{RESET}" )

        
        if os.path.isdir(matches[j]):
          
          if DEBUG>0:
            print ( f"CREATE_MASTER:     INFO:      {GREEN}directory {CYAN}{matches[j]}{RESET}{GREEN} exists{RESET}" )

          if DEBUG>9:        
            print ( f"CREATE_MASTER:     INFO:      directory {CYAN}{matches[j]}{RESET}" )                      
          found_clone_directories+=1
          
          for f in os.listdir(matches[j]):                                                                 # for each clone directory
            if f.endswith(".svs") or f.endswith(".SVS") or f.endswith(".tif") or f.endswith(".TIF"):
              found_slide_file                 +=1
              clone_found_slide_file           +=1
              global_found_slide_file          +=1
              found_file_of_interest           +=1
              clone_found_file_of_interest     +=1
              global_found_file_of_interest    +=1
    
              if DEBUG>0:
                print( f"CREATE_MASTER:     INFO:      in this dir:                  found slide file  {CARRIBEAN_GREEN}{f}{RESET}   number found = {CARRIBEAN_GREEN}{found_slide_file}{RESET}" )
              if DEBUG>11:
                print( f"CREATE_MASTER:     INFO:      {CARRIBEAN_GREEN}{df.iloc[i, clone_found_slide_file]}{RESET}" )
              
              df.iloc[i, image_column] = clone_found_slide_file
              
            elif f.endswith("FPKM-UQ.txt"):
              found_rna_seq_file               +=1
              clone_found_rna_seq_file         +=1
              global_found_rna_seq_file        +=1
              found_file_of_interest           +=1
              clone_found_file_of_interest     +=1
              global_found_file_of_interest    +=1

              if DEBUG>0:
                print( f"CREATE_MASTER:     INFO:      in this dir:          found rna-seq file  {BITTER_SWEET}{f}{RESET}   number found = {BITTER_SWEET}{found_rna_seq_file}{RESET}" )
              if DEBUG>11:
                print( f"CREATE_MASTER:     INFO:      in this dir: {BITTER_SWEET}{df.iloc[i, found_rna_seq_file]}{RESET}" )
              
              df.iloc[i, rna_seq_column] = clone_found_rna_seq_file          
              
            else:
              
              global_other_files_found+=1

            if found_slide_file>0 and found_rna_seq_file>0:
               matched_cases_count    +=1  
               if DEBUG>0:              
                 print( f"CREATE_MASTER:     INFO:        {MAGENTA}matched files{RESET}" ) 
            if DEBUG>0:
              if found_slide_file>1:
                print( f"CREATE_MASTER:     INFO:        {BLEU}multiple ({MIKADO}{found_slide_file}{RESET}) slide files exist in directory {CYAN}{matches[j]}{RESET}" ) 
            if DEBUG>0:
              if  found_rna_seq_file>1:
                print( f"CREATE_MASTER:     INFO:        {ORANGE}multiple ({MIKADO}{found_rna_seq_file}{RESET}{ORANGE}) rna-seq files exist in directory {CYAN}{matches[j]}{RESET}" ) 
            if DEBUG>0:                       
              if found_file_of_interest==0:
                print( f"CREATE_MASTER:     INFO:        {MAGENTA}no files of interest in directory {CYAN}{matches[j]}{RESET}" )  
              
    
          if DEBUG>9:
            print( f"CREATE_MASTER:     INFO:        clone dirs:        found   slide files {CARRIBEAN_GREEN}{clone_found_slide_file}{RESET}" )
            print( f"CREATE_MASTER:     INFO:        clone dirs:        found rna-seq files {BITTER_SWEET}{clone_found_rna_seq_file}{RESET}" )
          if DEBUG>11:
            print( f"CREATE_MASTER:     INFO:        clone dirs:        totals: {BITTER_SWEET}{df.iloc[i, clone_found_rna_seq_file]}{RESET}" )
          
        else:
          print ( f"CREATE_MASTER:     INFO:      {RED}directory {CYAN}{matches[j]}{RESET}{RED} does not exist{RESET}" )




  # (2) Cross check files in dataset against the applicable master spreadsheet
  
  actual_dirs=-1
  if DEBUG>0:                                                                                       # so that we don't count the root directory, only subdirectories
    print ( f"\nCREATE_MASTER:     INFO:    about to scan {CYAN}{class_specific_dataset_files_location}{RESET} to ensure all cases stored locally are also listed in the '{MAGENTA}{cancer_class}{RESET}' clinical master spreadsheet ('{CYAN}{master_spreadsheet}{RESET}'){RESET}" )

  for _, d, f in os.walk( class_specific_dataset_files_location ):
    actual_dirs+=1
    for el in enumerate ( d ):
      if DEBUG>9:
        print ( f"{PINK}length is {MIKADO}{len(case)}{RESET} {BLEU} and directory is {CYAN}{el[1]}{RESET}" )
      case_found_in_spreadsheet=False
      
      if re.search( "_[0-9]", el[1]):                                                                      # cases which have more than one RNA-seq example.  These have the extension _1 _2 etc.  Only cater for up to _9 coz never seen one with more than two
        if DEBUG>9:
          print ( (el[1])[:-2] )
      else:
        pass
      
      for c in range(2, len(df)):
        case =  df.iloc[c, 1]
        if DEBUG>99:
          print ( f"{BLEU}el {MIKADO}{el[1]}{RESET} {BLEU} against case {CYAN}{case}{RESET}" )  
        if ( el[1]==case ):
          case_found_in_spreadsheet=True
        if re.search( "_[0-9]", el[1] ):
          if (el[1])[:-2] == case:
            case_found_in_spreadsheet=True          
          
      if case_found_in_spreadsheet==True:
        if DEBUG>1:
          print ( f"{GREEN}directory (case) '{CYAN}{el[1]}{RESET}'{GREEN} \r\033[55C (or its root if applicable) is listed the applicable master clinical spreadsheet{RESET}" )
        else:
          pass
      else:
        print ( f"{RED}directory (case) '{CYAN}{el[1]}{RESET}{RED}'\r\033[62C(or its root if applicable) is not listed in master clinical spreadsheet\r\033[200C <<<<< anomoly, but no action will be taken{RESET}" )
    
    
  # (3) show some useful stats
  
  offset=176
  print ( f"\n" )    
  print ( f"CREATE_MASTER:     INFO:    total cases listed in TCGA {CYAN}{cancer_class}_global{RESET} master spreadsheet ('{CYAN}{master_spreadsheet}{RESET}') as edited:                                     \r\033[{offset}Cfound cases                                      =  {MIKADO}{found_cases}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total                  directories  (exc. clones) found in class specific dataset files location '{CYAN}{class_specific_dataset_files_location}{RESET}':              \r\033[{offset}Cfound (non_clone) directories                    =  {MIKADO}{found_non_clone_directories}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    {ITALICS}hence{RESET} total cases in master spreadsheet that don't exist in the local dataset:                                                            \r\033[{offset}Cfound_cases {BLEU}minus{RESET} found (non_clone) directories  =  {GREEN if found_cases-found_non_clone_directories==0 else RED}{found_cases-found_non_clone_directories:3d}{RESET}", end="" )
  if not found_cases - found_clone_directories == 0:
    print ( f"\r\033[235C{RED}  <<<<< this many cases don't exist in the class specific dataset files location{RESET}")
  else:
    print ("")
  print ( f"CREATE_MASTER:     INFO:    total examples  (clone directories) found in class specific dataset files location '{CYAN}{class_specific_dataset_files_location}{RESET}':                            \r\033[{offset}Cfound_clone_directories                          =  {MIKADO}{found_clone_directories}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total            clone directories        in class specific dataset files location '{CYAN}{class_specific_dataset_files_location}{RESET}'':                           \r\033[{offset}Cactual_dirs                                      =  {MIKADO}{actual_dirs}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    {ITALICS}hence{RESET}                  directories        in class specific dataset files location that don't correspond to a case in the master spreadsheet{RESET}': \r\033[{offset}Cactual_dirs {BLEU}minus{RESET} found_non_clone_directories  =  {GREEN if actual_dirs - found_clone_directories==0 else RED}{actual_dirs - found_clone_directories:2d}{RESET}", end="" )
  if not actual_dirs - found_clone_directories == 0:
    print ( f"\r\033[225C{RED}  <<<<< anomoly - not listed in spreadsheet{RESET}")
  else:
    print ("")
  print ( f"CREATE_MASTER:     INFO:    total {DIM_WHITE}files of no interest{RESET}   actually found  =  {DIM_WHITE}{global_other_files_found}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total {DIM_WHITE}files of    interest{RESET}   actually found  =  {DIM_WHITE}{global_found_file_of_interest}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total {CARRIBEAN_GREEN}slide{RESET}   files          actually found  =  {CARRIBEAN_GREEN}{global_found_slide_file}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total {BITTER_SWEET}rna-seq{RESET} files          actually found  =  {BITTER_SWEET}{global_found_rna_seq_file}{RESET}" )
  print ( f"CREATE_MASTER:     INFO:    total {BLEU}matched{RESET} cases                          =  {BLEU}{matched_cases_count}{RESET}" )
    

  save_file_name = f"{class_specific_global_data_location}/{dataset}_mapping_file_MASTER.csv"
  if (DEBUG>0):
    print ( f"\nCREATE_MASTER:     INFO:    about to save:   {MAGENTA}{save_file_name}{RESET}")
      
  try:
    df.to_csv( save_file_name, sep=',', index=False )
  except Exception as e:
    print ( f"{RED}CREATE_MASTER:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  could notw write {MAGENTA}{mapping_file_name}{RESET}{RED} ({MAGENTA}{local_cancer_specific_dataset}{RESET}{RED}){RESET}" )
    print ( f"{RED}CREATE_MASTER:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)  


  print( f"\nCREATE_MASTER:     INFO:   {MIKADO}finished{RESET}" )
  hours   = round((time.time() - start_time) / 3600, 1  )
  minutes = round((time.time() - start_time) / 60,   1  )
  seconds = round((time.time() - start_time), 0  )
  #pprint.log_section('Job complete in {:} mins'.format( minutes ) )

  print(f'CREATE_MASTER:     INFO:   took {MIKADO}{minutes}{RESET} mins ({MIKADO}{seconds:.1f}{RESET} secs)')
  


  
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--base_dir',                            type=str, default="/home/peter/git/pipeline"            )
    p.add_argument('--data_dir',                            type=str                                                )
    p.add_argument('--dataset',                             type=str,                            required=True      )
    p.add_argument('--mapping_file_name',                   type=str                                                )
    p.add_argument('--case_column',                         type=str, default="bcr_patient_uuid"                    )
    p.add_argument('--class_column',                        type=str, default="type_n"                              )

    args, _ = p.parse_known_args()


    main(args)
