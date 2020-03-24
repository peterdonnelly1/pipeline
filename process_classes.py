# Create files containing the class number (truth value) and save to a file in each case sub-directory( source of renaming information is MAPPING_FILE)
# This can also be used stand-alone - it won't affect data already downloaded into the data directory from GDC dir as it depends only on the mapping file
# The mapping file must be located at the top of the data directory
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

DEBUG=1

#====================================================================================================================================================
def main(args):
  
  data_dir              = args.data_dir
  mapping_file          = args.mapping_file
  case_column           = args.case_column
  class_column          = args.class_column
  class_numpy_filename  = args.class_numpy_filename
  
  if (DEBUG>0):
    print ( "PROCESS_CLASSES:        INFO: argv[1] (data_dir)             = {:}".format( data_dir             ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[2] (mapping_file)         = {:}".format( mapping_file         ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[3] (case_column)          = {:}".format( case_column          ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[4] (class_column)         = {:}".format( class_column         ),  flush=True )
    print ( "PROCESS_CLASSES:        INFO: argv[5] (class_numpy_filename) = {:}".format( class_numpy_filename ),  flush=True )
 
  all_classes=[]

  if (DEBUG>0):    
    print ( f"PROCESS_CLASSES:        INFO: about to open {mapping_file}")
  reader = csv.DictReader(open( mapping_file ))

  processed_count= 0
  tested_count   = 0
  
  for row in reader:


    RESET="\033[m"
    a = random.choice( range( 50,90) )
    b = random.choice( range( 50,90) )
    c = random.choice( range(100,130) )
    c = 120
    BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

    tested_count += 1

    if (DEBUG>0):    
      print ( "PROCESS_CLASSES:        INFO: row[case_column], row[class_column]     = {:}{:}{:}{:}".format( BB, row[case_column], row[class_column], RESET ) )

    case        =   row[case_column ]
    true_class  =   row[class_column]
    
    if (DEBUG>0):
      print ( "PROCESS_CLASSES:        INFO: processed_count   = {:}".format( processed_count ),  flush=True )
      print ( "PROCESS_CLASSES:        INFO: case id                                 = {:}{:}{:}".format( BB, case,  RESET ),  flush=True )

    target_dir =  "{:}/{:}*".format(  data_dir,  case  ) 
    if (DEBUG>0):
      print ( "PROCESS_CLASSES:        INFO: target_dir                              = {:}{:}{:}".format( BB, target_dir,    RESET ),  flush=True )

    found = []
    found = glob.glob( target_dir )  # returns an array holding a list of matches

    for d in found:                                                                                        # catering for cases where there are multiple images for the same case
      if (DEBUG>0):
        print ( "PROCESS_CLASSES:        INFO: dir                                     = {:}{:}{:}".format( BB, d, RESET ),  flush=True )
 
      if  os.path.exists( d  ):
        if (DEBUG>0):        
          print ( "PROCESS_CLASSES:        INFO: found directory                         = \033[32;1m{:}\033[m".format( d ),  flush=True )
          print ( "PROCESS_CLASSES:        INFO: class for this case                     = \033[32;1m{:}\033[m".format( true_class ),  flush=True )	
        
        tissue          = np.zeros( 1, dtype=int )
        tissue[0]       = true_class
        tissue_npy_file = os.path.join(data_dir, d, class_numpy_filename )
        if (DEBUG>0):        
          print ( "PROCESS_CLASSES:        INFO: about to save                            class value {:}{:}{:} to file {:}{:}{:} ".format( BB, tissue[0], RESET, BB, tissue_npy_file, RESET ),  flush=True )
        np.save(tissue_npy_file, tissue)
 
      processed_count+=1
      all_classes.append(true_class)

    if (DEBUG>0):
      print ( "PROCESS_CLASSES:        INFO: # of mapping file rows examined = \033[1m{:}\033[m".format ( tested_count ) )
      print ( "PROCESS_CLASSES:        INFO: # of class files created        = \033[1m{:}\033[m".format ( processed_count ) )
  
  all_classes_unique=sorted(set(all_classes))

  if (DEBUG>99):
    print ( f"PROCESS_CLASSES:        INFO: all class labels found (all_classes)    = \033[1m{all_classes}\033[m" )
    print ( f"PROCESS_CLASSES:        INFO: len class labels found len(all_classes) = \033[1m{len(all_classes)}\033[m" )
    print ( f"PROCESS_CLASSES:        INFO: unique classes represented              = \033[1m{all_classes_unique}\033[m" )
  
  as_integers = [int(i) for i in all_classes_unique]
  as_integers_sorted = sorted(as_integers)
  
  if (DEBUG>99):
    print ( f"{as_integers_sorted}" )
    print ( f"{ min (as_integers) }" )
    print ( f"{ max(as_integers)+1 }" )
    print ( f"{range(min(as_integers), max(as_integers)+1)}" )
  
  IsConsecutive= (sorted(as_integers) == list(range(min(as_integers), max(as_integers)+1)))
  if (DEBUG>0):
    print ( f"\033[32;1mPROCESS_CLASSES:        INFO: class labels consecutive                = {len(as_integers_sorted)}\033[m" )
    print ( f"\033[32;1mPROCESS_CLASSES:        INFO: number of truth labels                  = {IsConsecutive}\033[m" )
  
  if not IsConsecutive==True:
    print( f"\033[31;1mPROCESS_CLASSES:        FATAL: classes MUST start at be consecutive and start at zero. Halting now since training will fail\033[m" )
    print( f"\033[31;1mPROCESS_CLASSES:        FATAL: for reference, these are the classes that were found: {as_integers_sorted}\033[m" )
    sys.exit(0)
  
  degenerateClasses=(min(as_integers)<0)
  
  if degenerateClasses==True:
    print( f"\033[31;1m\033[1mPROCESS_CLASSES:        FATAL: classes MUST be integers zero or greater. Halting now since training will fail\033[m" )
    print( f"\033[31;1m\033[1mPROCESS_CLASSES:        FATAL: for reference, the lowest class value found was: {min(as_integers)}\033[m" )
    sys.exit(0)
  
  
    
  # now go through tree and delete any first level subfolder which does not contain a class.npy file (we can't use these)
    
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
          print ( "PROCESS_CLASSES:        INFO: this case did not obtain a class file; deleting: \033[31m{:}\033[m".format(   current_dir     ),  flush=True )    
        shutil.rmtree ( current_dir )

#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',              type=str, default="/home/peter/git/pipeline/dataset") 
  p.add_argument('--mapping_file',          type=str, default="./mapping_file") 
  p.add_argument('--class_numpy_filename',  type=str, default="class.npy") 
  p.add_argument('--case_column',           type=str, default="bcr_patient_uuid")
  p.add_argument('--class_column',          type=str, default="type_n")
    
#  p.add_argument('--case_column',        type=str, default="bcr_patient_uuid")
#  p.add_argument('--class_column',        type=str, default="bcr_patient_uuidhistological_type")
     
  args, _ = p.parse_known_args()

  main(args)
