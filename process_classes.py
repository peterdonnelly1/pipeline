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

from classi.constants  import *

DEBUG   = 1

#====================================================================================================================================================
def main(args):
  
  dataset                = args.dataset  
  data_dir               = args.data_dir
  global_data            = args.global_data  
  mapping_file           = args.mapping_file
  mapping_file_name      = args.mapping_file_name  
  case_column            = args.case_column
  class_column           = args.class_column
  class_numpy_filename   = args.class_numpy_filename
  skip_rna_preprocessing = args.skip_rna_preprocessing
  
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


  cases_processed_count  = 0
  cases_reviewed_count   = 0
  
  for row in reader:

    a = random.choice( range(  50,90 ) )
    b = random.choice( range(  50,90 ) )
    c = random.choice( range( 100,130) )
    c = 120
    BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

    cases_reviewed_count += 1

    if (DEBUG>8):    
      print ( f"PROCESS_CLASSES:        INFO: row[case_column] = {BLEU}{row[case_column]}{RESET}, row[class_column] = {MIKADO}{row[class_column]}{RESET}", flush=True )

    case        =   row[case_column ]
    true_class  =   row[class_column]
    
    if (DEBUG>9):
      print ( "PROCESS_CLASSES:        INFO: cases_processed_count   = {:}".format( cases_processed_count ),  flush=True )
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
 
      cases_processed_count+=1
      all_classes.append(true_class)

    if (DEBUG>9):
      print ( "PROCESS_CLASSES:        INFO: # of mapping file rows examined = \033[1m{:}\033[m".format ( cases_reviewed_count    ) )
      print ( "PROCESS_CLASSES:        INFO: # of class files created        = \033[1m{:}\033[m".format ( cases_processed_count ) )
  
    if (DEBUG>0):
      if cases_reviewed_count % 50==0:
        print ( f"PROCESS_RNA_EXP:        INFO: {MIKADO}{cases_reviewed_count}{RESET} cases listed; class (subtype) labels allocated to {MIKADO}{cases_processed_count}{RESET} RNA-Seq files in the working dataset directory accordingly {RESET}",  flush=True )
        print ( "\033[2A",  flush=True )

  if (DEBUG>0):                                                                                            # this will show the final count
    print ( f"PROCESS_RNA_EXP:        INFO: {MIKADO}{cases_reviewed_count}{RESET} cases reviewed; class (subtype) labels allocated to {MIKADO}{cases_processed_count}{RESET} RNA-Seq files in the working dataset directory accordingly {RESET}",  flush=True )
    print ( "\033[2A",  flush=True )

  if (DEBUG>0):
    print ( "\033[2B",  flush=True )
  
  
  all_classes_unique=sorted(set(all_classes))
  if (DEBUG>2):    
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
  
  cases_without_class_file_count = 0
  
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
        cases_without_class_file_count += 1
        if (DEBUG>2):
          print ( "PROCESS_CLASSES:        INFO: this case did not obtain a class file; deleting from working dataset: \033[31m{:}\033[m".format(   current_dir     ),  flush=True )    
        shutil.rmtree ( current_dir )

  if (DEBUG>0):
    if cases_without_class_file_count>0:
      print ( f"{BOLD}{ORANGE}PROCESS_CLASSES:        WARNG: {MIKADO}{cases_without_class_file_count}{RESET}{BOLD}{ORANGE} cases did not obtain a class file. Those cases have been deleted from the working dataset (but not from the source data directory){RESET}",  flush=True )    



#====================================================================================================================================================
      
if __name__ == '__main__':


  def str2bool(v):
      if isinstance(v, bool):
          return v
      if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
      elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
      else:
          raise argparse.ArgumentTypeError('Boolean value expected for this input parameter')
          
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',                type=str, default="/home/peter/git/pipeline/working_data"  )
  p.add_argument('--dataset',                 type=str                                              )
  p.add_argument('--global_data',             type=str                                              )  
  p.add_argument('--mapping_file',            type=str, default="./mapping_file"                    )
  p.add_argument('--mapping_file_name',       type=str, default="mapping_file"                      ) 
  p.add_argument('--class_numpy_filename',    type=str, default="class.npy"                         ) 
  p.add_argument('--case_column',             type=str, default="bcr_patient_uuid"                  )
  p.add_argument('--class_column',            type=str, default="type_n"                            )
  p.add_argument('--skip_rna_preprocessing',  type=str2bool, nargs='?', const=False, default=False, help="If true, don't preprocess RNA-Seq files")

#  p.add_argument('--case_column',        type=str, default="bcr_patient_uuid")
#  p.add_argument('--class_column',        type=str, default="bcr_patient_uuidhistological_type")
     
  args, _ = p.parse_known_args()

  main(args)
