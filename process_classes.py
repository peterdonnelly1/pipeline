# Create files containing the class number (truth value) and save to a file in each case sub-directory (source of renaming information is MAPPING_FILE)
# This can also be used stand-alone - it won't affect data already downloaded into the data directory from GDC dir as it depends only on the mapping file
# The mapping file must be located in the data directory
# Always strip the csv file extension off the mapping file name or else other scripts will confuse it with an RNA-seq file, which is also a csv file.

import os
import sys
import csv
import glob
import random
import fnmatch
import shutil
import pathlib
import argparse
import numpy  as np
import pandas as pd

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

#====================================================================================================================================================
def main(args):
  
  dataset               = args.dataset  
  data_dir              = args.data_dir
  global_data           = args.global_data  
  mapping_file          = args.mapping_file
  mapping_file_name     = args.mapping_file_name  
  case_column           = args.case_column
  class_column          = args.class_column
  class_numpy_filename  = args.class_numpy_filename
  
  if (DEBUG>1):
    print ( "PROCESS_CLASSES:        INFO: argv[1] (data_dir)             = {:}".format( data_dir             ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[2] (mapping_file)         = {:}".format( mapping_file         ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[3] (case_column)          = {:}".format( case_column          ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[4] (class_column)         = {:}".format( class_column         ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[5] (class_numpy_filename) = {:}".format( class_numpy_filename ),  flush=True )
 
  all_classes=[]

  if (DEBUG>0):    
    print ( f"PROCESS_CLASSES:        INFO: about to open:                           {MAGENTA}{mapping_file}{RESET}")

  try:
    reader = csv.DictReader(open( mapping_file ), delimiter=',')
  except Exception as e:
    print ( f"{RED}PROCESS_CLASSES:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}PROCESS_CLASSES:     FATAL:  explanation: there is no mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the dataset working copy ({MAGENTA}{data_dir}{RESET}{RED}){RESET}" )
    print ( f"{RED}PROCESS_CLASSES:     FATAL:  remedy: ensure there's a valid mapping file named {MAGENTA}{mapping_file_name}{RESET}{RED} in the {MAGENTA}{dataset}{RESET}{RED} source dataset directory ({MAGENTA}{global_data}{RESET}{RED}){RESET}" )                                   
    print ( f"{RED}PROCESS_CLASSES:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)  


  processed_count= 0
  tested_count   = 0
  
  for row in reader:

    a = random.choice( range(  50,90 ) )
    b = random.choice( range(  50,90 ) )
    c = random.choice( range( 100,130) )
    c = 120
    BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

    tested_count += 1

    if (DEBUG>8):    
      print ( f"PROCESS_CLASSES:        INFO: row[case_column] = {BLEU}{row[case_column]}{RESET}, row[class_column] = {MIKADO}{row[class_column]}{RESET}", flush=True )

    case        =   row[case_column ]
    true_class  =   row[class_column]
    
    if (DEBUG>9):
      print ( "PROCESS_CLASSES:        INFO: processed_count   = {:}".format( processed_count ),  flush=True )
      print ( "PROCESS_CLASSES:        INFO: case id                                 = {:}{:}{:}".format( BB, case,  RESET ),  flush=True )

    target_dir =  "{:}/{:}*".format(  data_dir,  case  ) 
    if (DEBUG>9):
      print ( "PROCESS_CLASSES:        INFO: target_dir                              = {:}{:}{:}".format( BB, target_dir,    RESET ),  flush=True )

    found = []
    found = glob.glob( target_dir )  # returns an array holding a list of matches

    for d in found:                                                                                        # catering for cases where there are multiple images for the same case
      if (DEBUG>9):
        print ( "PROCESS_CLASSES:        INFO: dir                                     = {:}{:}{:}".format( BB, d, RESET ),  flush=True )
 
      if  os.path.exists( d  ):
        if (DEBUG>2):        
          print ( f"PROCESS_CLASSES:        INFO: found directory: {CYAN}{d}{RESET} \r\033[130Cand class for this case = {MIKADO}{true_class}{RESET}",  flush=True )
        
        tissue          = np.zeros( 1, dtype=int )
        tissue[0]       = true_class
        tissue_npy_file = os.path.join(data_dir, d, class_numpy_filename )
        if (DEBUG>8):        
          print ( f"PROCESS_CLASSES:        INFO: about to save                            class value {MIKADO}{tissue[0]}{RESET} to file {BLEU}{tissue_npy_file}{RESET}",  flush=True )
        np.save(tissue_npy_file, tissue)
 
      processed_count+=1
      all_classes.append(true_class)

    if (DEBUG>9):
      print ( "PROCESS_CLASSES:        INFO: # of mapping file rows examined = \033[1m{:}\033[m".format ( tested_count ) )
      print ( "PROCESS_CLASSES:        INFO: # of class files created        = \033[1m{:}\033[m".format ( processed_count ) )
  
  all_classes_unique=sorted(set(all_classes))
  if (DEBUG>0):    
    print ( f"{DIM_WHITE}PROCESS_CLASSES:        INFO: unique subtypes seen in dataset       = {MIKADO}{all_classes_unique}{RESET}   {DIM_WHITE}CAUTION! for a given dataset (e.g. stad), a given subtype may have image examples but not RNA-Seq examples, or vice-versa{RESET}" )
    
  if (DEBUG>99):
    print ( f"PROCESS_CLASSES:        INFO: all class labels found (as determined from dataset, not MASTER spreadsheet)    = \033[1m{all_classes}\033[m" )
    print ( f"PROCESS_CLASSES:        INFO: len class labels found (as determined from dataset, not MASTER spreadsheet) = \033[1m{len(all_classes)}\033[m" )

  
  as_integers = [int(i) for i in all_classes_unique]
  as_integers_sorted = sorted( as_integers )
  
  if (DEBUG>99):
    print ( f"{as_integers_sorted}"   )
    print ( f"{ min (as_integers) }"  )
    print ( f"{ max(as_integers)+1 }" )
    print ( f"{range(min(as_integers), max(as_integers)+1)}" )
  
  IsConsecutive= (sorted(as_integers) == list(range(min(as_integers), max(as_integers)+1)))
  if (DEBUG>0):
    print ( f"{DIM_WHITE}PROCESS_CLASSES:        INFO: number of classes observed            = {MIKADO}{len(as_integers_sorted)}{RESET}" )
    print ( f"{DIM_WHITE}PROCESS_CLASSES:        INFO: class labels consecutive?             = {MIKADO}{IsConsecutive}{RESET}" )
  
  if not IsConsecutive==True:
    print( f"\033[31;1mPROCESS_CLASSES:        FATAL: classes MUST start at be consecutive and start at zero. Halting now since training will fail\033[m" )
    print( f"\033[31;1mPROCESS_CLASSES:        FATAL: for reference, these are the classes that were found: {as_integers_sorted}\033[m" )
    sys.exit(0)
  
  degenerateClasses=(min(as_integers)<0)
  
  if degenerateClasses==True:
    print( f"\033[31;1m\033[1mPROCESS_CLASSES:        FATAL: classes MUST be integers zero or greater. Halting now since training will fail\033[m" )
    print( f"\033[31;1m\033[1mPROCESS_CLASSES:        FATAL: for reference, the lowest class value found was: {min(as_integers)}\033[m" )
    sys.exit(0)
  
  
    
  # now go through tree and delete any first level subfolder which does not contain a class.npy file (we can't use these). By doing this we exclude the sample from the experiment
    
  walker = os.walk( data_dir )
  for root, dirs, files in walker:
    for d in dirs:
      current_dir    = os.path.join( root, d )
      if (DEBUG>99):
        print ( current_dir )
      has_class_file=False
      for f in os.listdir(current_dir):
        if class_numpy_filename in f:
          if (DEBUG>99):
            print(f)
          has_class_file=True
      
      if has_class_file==False:
        if (DEBUG>0):
          print ( "PROCESS_CLASSES:        INFO: this case did not obtain a class file; deleting from working dataset: \033[31m{:}\033[m".format(   current_dir     ),  flush=True )    
        shutil.rmtree ( current_dir )

#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',              type=str, default="/home/peter/git/pipeline/dataset"  )
  p.add_argument('--dataset',               type=str                                              )
  p.add_argument('--global_data',           type=str                                              )  
  p.add_argument('--mapping_file',          type=str, default="./mapping_file"                    )
  p.add_argument('--mapping_file_name',     type=str, default="mapping_file"                      ) 
  p.add_argument('--class_numpy_filename',  type=str, default="class.npy"                         ) 
  p.add_argument('--case_column',           type=str, default="bcr_patient_uuid"                  )
  p.add_argument('--class_column',          type=str, default="type_n"                            )
    
#  p.add_argument('--case_column',        type=str, default="bcr_patient_uuid")
#  p.add_argument('--class_column',        type=str, default="bcr_patient_uuidhistological_type")
     
  args, _ = p.parse_known_args()

  main(args)
