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
import argparse
import pathlib
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
    print ( "    PROCESS_CLASSES.PY: INFO: argv[1] (data_dir)             = {:}".format( data_dir             ),  flush=True )
    print ( "    PROCESS_CLASSES.PY: INFO: argv[2] (mapping_file)         = {:}".format( mapping_file         ),  flush=True )
    print ( "    PROCESS_CLASSES.PY: INFO: argv[3] (case_column)          = {:}".format( case_column          ),  flush=True )
    print ( "    PROCESS_CLASSES.PY: INFO: argv[4] (class_column)         = {:}".format( class_column         ),  flush=True )
    print ( "    PROCESS_CLASSES.PY: INFO: argv[5] (class_numpy_filename) = {:}".format( class_numpy_filename ),  flush=True )

  reader = csv.DictReader(open( mapping_file ))

  count=0

  for row in reader:

    RESET="\033[m"
    a = random.choice( range( 50,90) )
    b = random.choice( range( 50,90) )
    c = random.choice( range(100,130) )
    c = 120
    BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

    if (DEBUG>1):
      print ( "    PROCESS_CLASSES.PY: INFO: row[case_column], row[class_column] = {:}{:}{:}{:}".format( BB, row[case_column], row[class_column], RESET ) )

    case        =   row[case_column ]
    true_class  =   row[class_column]
    
    if (DEBUG>0):
      print ( "    PROCESS_CLASSES.PY:     INFO: count                   = {:}{:}{:}".format( BB, count, RESET ),  flush=True )
      print ( "    PROCESS_CLASSES.PY:     INFO: case id                 = {:}{:}{:}".format( BB, case,  RESET ),  flush=True )

    target_dir =  "{:}/{:}*".format(  data_dir,  case  ) 
    if (DEBUG>0):
      print ( "      PROCESS_CLASSES.PY:   INFO: target_dir              = {:}{:}{:}".format( BB, target_dir,    RESET ),  flush=True )

    found = glob.glob( target_dir )
    
    for d in found:
      if (DEBUG>0):
        print ( "        PROCESS_CLASSES.PY: INFO: dir                     = {:}{:}{:}".format( BB, d, RESET ),  flush=True )
 
      if  os.path.exists( d  ):
        if (DEBUG>0):        
          print ( "        PROCESS_CLASSES.PY: INFO: found directory         = \033[32;1m{:}\033[m".format( d ),  flush=True )
          print ( "        PROCESS_CLASSES.PY: INFO: class for this case     = \033[32;1m{:}\033[m".format( true_class ),  flush=True )	
        
        tissue          = np.zeros( 1, dtype=int )
        tissue[0]       = true_class
        tissue_npy_file = os.path.join(data_dir, d, class_numpy_filename )
        if (DEBUG>0):        
          print ( "        PROCESS_CLASSES.PY: INFO: about to save             class value {:}{:}{:} to file {:}{:}{:} ".format( BB, tissue[0], RESET, BB, tissue_npy_file, RESET ),  flush=True )
        np.save(tissue_npy_file, tissue)
        
      else:
        print ( "        PROCESS_CLASSES.PY: INFO:                                    \033[31;1mno directory found for that case\033[m".format ( case ) )

      count+=1

	  print ( "    PROCESS_CLASSES.PY: INFO: # of class files created = \033[1m{:}\033[m".format ( count ) )     

#====================================================================================================================================================
      
if __name__ == '__main__':
	
  p = argparse.ArgumentParser()

  p.add_argument('--data_dir',              type=str, default="/home/peter/git/pipeline/data") 
  p.add_argument('--mapping_file',          type=str, default="./mapping_file_no_dups") 
  p.add_argument('--class_numpy_filename',  type=str, default="class.npy") 
  p.add_argument('--case_column',           type=str, default="tcga_entity_id")
  p.add_argument('--class_column',          type=str, default="type_n")
    
#  p.add_argument('--case_column',        type=str, default="bcr_patient_uuid")
#  p.add_argument('--class_column',        type=str, default="bcr_patient_uuidhistological_type")
     
  args, _ = p.parse_known_args()

  main(args)
