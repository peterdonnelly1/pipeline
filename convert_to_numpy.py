"""
open each rna results file within the patches directory; extract the rna-seq "scaled estimate" column; convert to numpy array; save numpy array as a new file with the same name but .npy rna_file_pattern

"""

import os
import sys
import codecs
import random
import fnmatch
import numpy  as np
import pandas as pd

DEBUG=1
RESET="\033[m"
a = random.choice( range(200,255) )
b = random.choice( range(50,225) )
c = random.choice( range(50,225) )
BB="\033[38;2;{:};{:};{:}m".format( a,b,c )

cumulative_rna_results        = 0
cumulative_tissue_class_count = 0

files_dir              = sys.argv[1]
rna_file_pattern       = sys.argv[2]
rna_numpy_filename     = sys.argv[3]
tissue_numpy_filename  = sys.argv[4]
tissue_file_extension  = "csv"

if (DEBUG>0):
  print ( "    CONVERT_TO_NUMPY.PY: INFO: argv[1] (files_dir)             = {:}{:}{:}".format( BB, files_dir,             RESET ),  flush=True )
  print ( "    CONVERT_TO_NUMPY.PY: INFO: argv[2] (rna_file_pattern)      = {:}{:}{:}".format( BB, rna_file_pattern,      RESET ),  flush=True )
  print ( "    CONVERT_TO_NUMPY.PY: INFO: argv[3] (rna_numpy_filename)    = {:}{:}{:}".format( BB, rna_numpy_filename,    RESET ),  flush=True )
  print ( "    CONVERT_TO_NUMPY.PY: INFO: argv[4] (tissue_numpy_filename) = {:}{:}{:}".format( BB, tissue_numpy_filename, RESET ),  flush=True )
 
walker = os.walk(files_dir)
for root, __, files in walker:
  for f in files:
    current_file    = os.path.join( root, f)
    rna_npy_file    = os.path.join( root, rna_numpy_filename )

    if (DEBUG>9): 
      print ( "    CONVERT_TO_NUMPY.PY: INFO: (rna_npy_file)                  = {:}{:}{:}".format( BB, rna_npy_file, RESET ),  flush=True )  

    # (1) Handle RNA data
    if fnmatch.fnmatch(f, rna_file_pattern ):
 
      rna_results_file_found   =1
      cumulative_rna_results  +=1  
      
      if (DEBUG>0): 
        print ( "    CONVERT_TO_NUMPY.PY: INFO: (match !)                         {:}{:}{:}   cumulative match count =".format( BB, current_file, RESET, BB, cumulative_rna_results, RESET ),  flush=True )
        print ( "    CONVERT_TO_NUMPY.PY: INFO: (rna_npy_file)                    {:}{:}{:}".format( BB, rna_npy_file, RESET ),  flush=True )
                
      rna_expression_column = pd.read_csv(current_file, sep='\t', usecols=[1])
      
      if DEBUG>1:
        print ( "CONVERT_TO_NUMPY: rna_expression_column as pandas object = \n\033[35m{:}\033[m".format(rna_expression_column[0:12]))
      
      rna = rna_expression_column.to_numpy()
  
      if DEBUG>1:
        print ( "CONVERT_TO_NUMPY: rna_expression_column as numpy array   = \n\033[35m{:}\033[m".format(rna[0:12]))
      
      np.save(rna_npy_file, rna)

    # (2) Handle tissue type data
    
    tissue_npy_file = os.path.join(root, tissue_numpy_filename )
    
    if ( f.endswith('.' + tissue_file_extension) ):                                              # expect to find
  
      tissue_class_file_found        =1
      cumulative_tissue_class_count +=1
  
      if DEBUG>0:
        print ( "CONVERT_TO_NUMPY: (\033[33m{:}\033[m) file name: \033[33m{:}\033[m".format(cumulative_tissue_class_count, current_file))
  
      tissue = np.array( [171], dtype=int )                                                              # any old distinctive number for debugging purposes
      
      #if DEBUG>0:
       # print ( "CONVERT_TO_NUMPY: tissue.shape  = \033[35;1m{:}\033[m".format(tissue.shape ) )
        #print ( "CONVERT_TO_NUMPY: tissue        = \033[35;1m{:}\033[m".format(tissue       ) ) 
        
      tissue[0]=np.genfromtxt(current_file, dtype=None)
      
      if DEBUG>9:
        print ( "CONVERT_TO_NUMPY: type(tissue[0] = \033[35;1m{:}\033[m".format(type(tissue[0]  ) ) )
        print ( "CONVERT_TO_NUMPY: tissue.shape   = \033[35;1m{:}\033[m".format(tissue.shape    ) )
        print ( "CONVERT_TO_NUMPY: tissue         = \033[35;1m{:}\033[m".format(tissue          ) )
        print ( "CONVERT_TO_NUMPY: tissue[0]      = \033[35;1m{:}\033[m".format(tissue[0]       ) )
      
      np.save(tissue_npy_file, tissue)

