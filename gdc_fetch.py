import os
import re
import json
import gzip
import glob
import fnmatch
import requests
import random
import argparse
import tarfile
import shutil as sh
from   pathlib  import Path

SUCCESS = 1

def main(args):
  
  DEBUG            = args.debug
  output_dir       = args.output_dir
  overlay          = args.overlay
  case_filter      = args.case_filter 
  file_filter      = args.file_filter
  portal           = args.gdc_portal
	  
  if(os.path.isdir( output_dir )):
    user_input = input( "\033[1mWARNING: directory named \033[31;1;4m{:}\033[m\033[1m already exists, perhaps from previous interrupted run. \033[31;1;4mc\033[m\033[1momplete previous download or \033[31;1;4mo\033[m\033[1mverlay new data\033[m or \033[31;1;4md\033[m\033[1melete directory and start afresh?  \033[m".format(output_dir) )
  
    while True:
      if user_input=='c':
    	  break
      elif user_input=='d':
        try:
          sh.rmtree(output_dir)
        except OSError:
          pass
        os.makedcase_files,irs(output_dir)
        break
      elif user_input=='o':
        overlay=1
        break
      else:
        print ("sorry, that's not an available option" )
        exit(0)

  else:
    pass

###########################################################################################################################################
# STEP 1: RETRIEVE CASE UUIDs OF CASES WHICH MEET SEARCH CRITERIA PROVIDED TO THE GDC API
###########################################################################################################################################

  if portal == "main":
    cases_endpt = "https://api.gdc.cancer.gov/cases"
  elif portal == "legacy":
    cases_endpt = "https://api.gdc.cancer.gov/legacy/cases"
  else:
    print( "\nGDC_FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )



  if DEBUG>0:
    print( "\nGDC_FETCH:  \033[1mSTEP 1:\033[m about to retrieve case UUIDs of cases which meet the provided search criteria" )
  
  fields = [
      "case_id"
      ]
  
  fields = ",".join(fields)


  with open( case_filter, 'r') as file:
    filters = json.load(file)

  # With a GET request, the filters parameter needs to be converted from a dictionary to JSON-formatted string
  params1 = {
      "filters": json.dumps(filters),
      "fields": fields,
      "format": "JSON",
      "size": args.max_cases
      }
  
  response = requests.get(cases_endpt, params=params1)
  
  cases_uuid_list = []
  
  for case_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
      cases_uuid_list.append(case_entry["case_id"])
  
  if DEBUG>0:
    print( "GDC_FETCH:  response (should be a json struct of the fields we requested. We are only interested in 'case_id') = {:}\033[m".format( response.text ) )
    print( "GDC_FETCH:  cases_uuid_list = \033[36;1m{:}\033[m".format( cases_uuid_list) )


  if DEBUG>0:
    print( "GDC_FETCH:  \033[1mSTEP 2: about to loop through each case UUID and request the UUIDs of associated files for each case\033[m" )


###########################################################################################################################################
# STEP 2: LOOP THROUGH EACH CASE ID IN cases_uuid_list AND PROCESS
# 
# Pseudocode:
#    for each case:
#      2a  fetch file ids for files of interest                                                       - fetch_case_file_ids()
#      2b  download these files                                                                       - download()
#      2c  unpack the tarball then delete the tarball                                                 - unpack_tarball()
#            tar will put decompressed files into the same directory which is where we want them
#            SVS files will not be further compressed
#            some other files including RNA-SEQ files will still be compressed as .gz files
#      2d  further decompress any gz files revealed                                                   - decompress_gz_files
#              gzip places decompressed gz files into a subdirectory of the case id directory
#      2e  promote leaf files to case id subdirectory                                                 - promote_leaf_files()
#            i.e. decompressed .gz files
#            at this point we have all wanted files at the case id level
#      2f  set up and populate a new case_id subdirectory for each SVS file downloaded                - setup_and_fill_case_subdirs()
#            for each SVS file (n, SVS file)
#              make a new subdirectory at the case id level with the extension case_id-<n>             
#              copy the SVS file plus the RNA-SEQ file into the new subdirectory
#      2g  delete the original case_id directory                                                      - delete_orignal_files()
#      2h  create a new case level subdirectory named to flag that the case was handled successfully  - _all_downloaded_ok()
#            checked on subsequent runs of gdc_fetch, so that files are not needlessly re-downloaded
#            especially SVS files, which can be extremely large (multi-gigabyte)
#  
###########################################################################################################################################
  
  n=0
  
  for case in cases_uuid_list:	
    
    a = random.choice( range(150,230) )
    b = random.choice( range(200,235) )
    c = random.choice( range(150,230) )
    RC="\033[38;2;{:};{:};{:}m".format( a,b,c )
  
    n+=1
    
    case_path = "{:}/{:}/".format( output_dir, case )

    if DEBUG>0:
      print( "\nGDC_FETCH:    case {:}{:}\033[m of {:}{:}\033[m".format( RC, n, RC, len( cases_uuid_list) ) )

    if DEBUG>0:
      print( "GDC_FETCH:    case id \033[1m{:}{:}\033[m".format( RC, case ) )

    already_have_flag = case_path[:-1] +  '_all_downloaded_ok'
    if DEBUG>9:
      print( "GDC_FETCH:    'already_have_flag'                             =  {:}{:}\033[m".format( RC,  already_have_flag ) )
      
    if ( overlay==0 ) & ( Path( already_have_flag ).is_dir()):                                          # Id the files for this case were already previously downloaded, then move to next case
        print( "GDC_FETCH:    \033[1m2a:\033[m files already exist for case                =        {:}{:} \033[m                    ... skipping and moving to next case\033[m".format( RC, case ) )

    else:
      if DEBUG>0:
        print( "GDC_FETCH:    \033[1m2a:\033[m requesting file UUIDs for case                 {:}{:}\033[m".format( RC, case ) )

      case_files = fetch_case_file_ids            ( RC, DEBUG,            case,         portal,  file_filter                     )
      SINGLETON_DOWNLOAD, tarfile_name = download ( RC, DEBUG, case_path, case_files,   portal                                   )
      result = unpack_tarball                     ( RC, DEBUG, case_path, tarfile_name,                     SINGLETON_DOWNLOAD   )
      result = decompress_gz_files                ( RC, DEBUG, case_path                                                         )
      result = promote_leaf_files                 ( RC, DEBUG, case_path,                                   SINGLETON_DOWNLOAD   )
      result = setup_and_fill_case_subdirs        ( RC, DEBUG, case_path                                                         )
      result = delete_orignal_files               ( RC, DEBUG, case_path,                                   SINGLETON_DOWNLOAD   )
      result = _all_downloaded_ok                 ( RC, DEBUG, case_path                                                         )    

    if DEBUG>0:
      print( "\nGDC_FETCH:    all done".format )

#====================================================================================================================================================
# 2a FETCH FILE IDs

def fetch_case_file_ids( RC, DEBUG, case, portal,  file_filter ):

  if portal == "main":
    files_endpt = "https://api.gdc.cancer.gov/files"
  elif portal == "legacy":
    files_endpt = "https://api.gdc.cancer.gov/legacy/files"
  else:
    print( "\nGDC_FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )

  
  with open( file_filter, 'r') as file:
   filters = json.load(file)
 
   if DEBUG>99:
     print ( filters['content'][0]['content']['value'] )
   
   filters['content'][0]['content']['value']  = case

   if DEBUG>99:
     print ( filters['content'][0]['content']['value'] )
  
  params2 = {
      "filters": json.dumps(filters),
      "fields": "file_id",
      "format": "JSON",
      "size": args.max_files
      }

  case_files = requests.get(files_endpt, params=params2)
  
  if DEBUG>9:
    print( "GDC_FETCH:          response (json list of file ids of hits)  =  {:}{:}\033[m".format(RC, case_files.text ) )
  
  return case_files

#====================================================================================================================================================
# 2b DOWNLOAD CASE FILES

def download( RC, DEBUG, case_path, case_files, portal ):
    
  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2b:\033[m about to populate file UUID download list and request files" )
    
  file_uuid_list = []

  # (i) Populate the download list with the file_ids from the previous query
  for file_entry in json.loads(case_files.content.decode("utf-8"))["data"]["hits"]:

    file_uuid_list.append(file_entry["file_id"])
  
  SINGLETON_DOWNLOAD=False
  if len(file_uuid_list)==1:
    SINGLETON_DOWNLOAD=True
      
  if DEBUG>0:
    print( "GDC_FETCH:          file_uuid_list (list of just file uuids)    =  {:}{:}\033[m".format( RC, file_uuid_list) )


  # (ii) Request, download and save the files (there will only ever be ONE actual file downloded because the GDC portal will put multiple files into a tar archive)

  if portal == "main":
    data_endpt = "https://api.gdc.cancer.gov/data"
  elif portal == "legacy":
    data_endpt = "https://api.gdc.cancer.gov/legacy/data"  
  else:
    print( "\nGDC_FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )
  
  params = {"ids": file_uuid_list}
  
  response = requests.post( data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})
  
  response_head_cd = response.headers["Content-Disposition"]

  if DEBUG>9:
    print( "GDC_FETCH:          response.headers[Content-Type             = {:}'{:}'\033[m".format( RC, response_head_cd ) )
  
  download_file_name = re.findall("filename=(.+)", response_head_cd)[0]                                            # extract filename from HTTP response header
 
  if DEBUG>9:
    print( "GDC_FETCH:          response.headers[Content-Disposition]     = {:}'{:}'\033[m".format( RC, download_file_name ) )
    print( "GDC_FETCH:          download_file_subdir_name                 = {:}'{:}'\033[m".format( RC, case_path ) )

  if not os.path.isdir ( case_path ):
    os.makedirs( case_path )
  
  download_file_fq_name = "{:}/{:}".format( case_path, download_file_name )

  if DEBUG>9:
    print( "GDC_FETCH:          download_file_fq_name                     = {:}'{:}'\033[m".format( RC, download_file_fq_name ) )

  with open(download_file_fq_name, "wb") as output_file_handle:
    output_file_handle.write(response	.content)

  return SINGLETON_DOWNLOAD, download_file_name
  
#====================================================================================================================================================
# 2c UNPACK TARBALL
 
def unpack_tarball ( RC, DEBUG, case_path, tarfile_name, SINGLETON_DOWNLOAD ):
  
  if not SINGLETON_DOWNLOAD:
    if DEBUG>0:
      print( "GDC_FETCH:    \033[1m2c:\033[m about to unpack tarball                     {:}'{:}'\033[m  to  {:}'{:}'\033[m".format( RC, tarfile_name, RC, case_path ) )
  else:
    if DEBUG>0:
      print( "GDC_FETCH:    \033[1m2c:\033[m no tarball to unpack. must be singleton\033[m" )
    return

  try:
    tarfile_fq_name = "{:}/{:}".format( case_path, tarfile_name )   
    tar = tarfile.open( tarfile_fq_name )
  except Exception:
    pass

  try:
    tar = tar.extractall( path=case_path )
  except Exception:
    pass

  try:
    tar.close()
  except Exception:
    pass

  if DEBUG>0:
    print( "GDC_FETCH:          tarball unpacked Ok\033[m" )

  try:
    os.remove( tarfile_fq_name )
    if DEBUG>0:
      print( "GDC_FETCH:    \033[1m2c:\033[m now deleting tarball" )
  except Exception:
    pass

  return SUCCESS

#====================================================================================================================================================
# 2d DECOMPRESS ANY GZ FILES WHICH MAY HAVE BEEN DOWNLOADED

def decompress_gz_files( RC, DEBUG, case_path ):

  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2d:\033[m unzipping all gz files in case path           {:}'{:}'\033[m, using match pattern {:}'{:}*.gz'\033[m".format( RC, case_path, RC, case_path ) )
    
  walker = os.walk( case_path )
  for root, _, files in walker:
    for gz_candidate in files:
      if  ( ( fnmatch.fnmatch( gz_candidate,"*.gz") )  ):                                                  # if it's a gz file

        if DEBUG>9:
          print( "GDC_FETCH:             opening                                  {:}'{:}'\033[m".format( RC, gz_candidate ) )
    
        fq_name = "{:}/{:}".format( root, gz_candidate )
        with gzip.open( fq_name, 'rb') as f:
          s = f.read()
            
        output_name    = fq_name[:-3]                                                                      # remove '.gz' extension from the filename
      
        if DEBUG>9:
          print( "GDC_FETCH:             saving decompressed file as              {:}'{:}'\033[m".format( RC, output_name ) )
    
        with open(output_name, 'wb') as f:                                                                 # store uncompressed data
          f.write(s)
        f.close

  return SUCCESS
  
#====================================================================================================================================================
# 2e PROMOTE LEAF FILES UP INTO CASE DIRECTORY

def promote_leaf_files( RC, DEBUG, case_path, SINGLETON_DOWNLOAD ):

  if not SINGLETON_DOWNLOAD:
    if DEBUG>0:
      print( "GDC_FETCH:    \033[1m2e:\033[m about to promote leaf files of interest up into the case directory" )
  else:
    if DEBUG>0:
      print( "GDC_FETCH:    \033[1m2e:\033[m singleton download. Will not promote leaf files" )
    return

  walker = os.walk( case_path, topdown=True )
  for root, _, files in walker:

    for f in files:

      fq_name = "{:}{:}".format( case_path, f )
                                                                          
      if  ( ( fnmatch.fnmatch( f, "MANIFEST.*" ) )  ):                                                     # have to delete because "MANIFEST.TXT" always only one level down, not two levels down
        if DEBUG>0:
          print ( "GDC_FETCH:          about to delete                           = {:}'{:}'\033[m".format( RC, fq_name )      )
        try:
          os.remove( fq_name )
        except Exception:
          pass
        
      elif (  ( f.endswith(".svs")  )   | ( f.endswith(".txt") ) ): 
        fq_name      = "{:}/{:}".format( root, f )
        if DEBUG>0:
          print ( "GDC_FETCH:          moving file up a level: filename          = {:}'{:}'\033[m".format( RC, fq_name )      )
        move_to_name = "{:}/../{:}".format( root, f )
        if DEBUG>0:
          print ( "GDC_FETCH:          moving file up a level: new name          = {:}'{:}\033[m".format( RC, move_to_name ) )     
  
        sh.move( fq_name, move_to_name )

      else:
        pass

  return SUCCESS
 
#====================================================================================================================================================
# 2f SET UP AND POPULATE A NEW CASE_ID SUBDIRECTORY FOR EACH SVS FILE

#            for each SVS file (n, SVS file)
#              make a new subdirectory at the case id level with the extension case_id-<n>             
#              copy the SVS file plus the RNA-SEQ file into the new subdirectory

def setup_and_fill_case_subdirs    ( RC, DEBUG, case_path ):

  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2f:\033[m about to set up and populate a new case_id subdirectory for each svs file found" )
    
    svs_count   = 0
    other_count = 0

    for f in os.listdir( case_path ):
      if f.endswith(".svs"):
        svs_count+=1
        
        new_dir_name = case_path[:-1] + '_' + str( svs_count )
        
        if DEBUG>0:
          print( "GDC_FETCH:          about to create new directory             '{:}{:}'\033[m".format        ( RC, new_dir_name           ) )
        new_dir = os.mkdir( new_dir_name )                                                                 # create a new case-level subdirectory with unique numeric suffix for this SVS file

        if DEBUG>0:
          print( "GDC_FETCH:          SVS   file count = {:}{:} and file name is     '{:}{:}'\033[m".format   ( RC, svs_count, RC, f       ) )        
          print( "GDC_FETCH:            about to move SVS file to new directory '{:}{:}' ".format             ( RC, new_dir_name           ) )
        existing_SVS_FQ_name = str(   case_path  )       +           str(f)
        if DEBUG>0:
          print( "GDC_FETCH:            old FQ name =                           '{:}{:}'\033[m".format        ( RC, existing_SVS_FQ_name   ) )
        new_SVS_FQ_name      = str( new_dir_name ) + '/' +           str(f)
        if DEBUG>0:
          print( "GDC_FETCH:            new FQ name =                           '{:}{:}'\033[m".format        ( RC, new_SVS_FQ_name        ) )
        os.rename(   existing_SVS_FQ_name, new_SVS_FQ_name     )
    
    for f in os.listdir( case_path ):
      if f.endswith(".txt"):
        other_count+=1
        
        for n in range( 0, svs_count ) :                                                                #         copy it into each of the 'svs_count' new folders created just above
        
          target_dir_name = case_path[:-1] + '_' + str( n+1 )
          
          if DEBUG>0:
            print( "GDC_FETCH:          n = {:}{:}\033[m of {:}{:}\033[m".format                              ( RC, n, RC, svs_count       ) )
            print( "GDC_FETCH:          OTHER file count = {:}{:} and file name is     '{:}{:}'\033[m".format ( RC, other_count, RC, f     ) )
            print( "GDC_FETCH:            about to copy file to new directory     '{:}{:}'\033[m".format      ( RC, target_dir_name        ) )
          if f.endswith(".txt"):
            existing_OTHER_FQ_name = str( case_path )       +        str(f)
          if DEBUG>0:
            print( "GDC_FETCH:            old FQ name =                           '{:}{:}'\033[m".format      ( RC, existing_OTHER_FQ_name ) )
          new_other_FQ_name        = str( target_dir_name ) + '/' +  str(f)
          if DEBUG>0:
            print( "GDC_FETCH:            new FQ name =                           '{:}{:}'\033[m".format      ( RC, new_other_FQ_name      ) )		  
          sh.copyfile( existing_OTHER_FQ_name, new_other_FQ_name )

      else:
        if DEBUG>0: 
          print( "GDC_FETCH:          this file will be ignored                   '{:}{:}'\033[m".format       ( RC, f                     ) )
      

#====================================================================================================================================================
# 2g DELETE ALL FILES IN THE ORIGINAL CASE_ID DIRECTORY

def delete_orignal_files( RC, DEBUG, case_path, SINGLETON_DOWNLOAD):
	
  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2g:\033[m about to delete all files in the original case_id directory" )
    
  if not SINGLETON_DOWNLOAD:    
    sh.rmtree( case_path )
  
  #  if SINGLETON_DOWNLOAD:
  #     just delete gz files and leave everythine else

  return SUCCESS

#====================================================================================================================================================
#  2h  create a new case level subdirectory named to indicate the case was handled successfully

def _all_downloaded_ok( RC, DEBUG, case_path ):

  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2h:\033[m about to create new case level subdirectory named to indicate the case was handled successfully" )
    
    if not os.path.isdir ( case_path ):
      os.mkdir( case_path[:-1] + '_all_downloaded_ok' )

  return SUCCESS
  
#====================================================================================================================================================
      
if __name__ == '__main__':
	
    p = argparse.ArgumentParser()

    p.add_argument('--debug',              type=int, default=1)
    p.add_argument('--gdc_portal',         type=str, default="main")
    p.add_argument('--case_filter',        type=str, default="dlbc_case_filter")
    p.add_argument('--file_filter',        type=str, default="dlbc_file_filter_just_rna-seq")
    p.add_argument('--max_cases',          type=int, default=5)
    p.add_argument('--max_files',          type=int, default=10)
    p.add_argument('--output_dir',         type=str, default="out")
    p.add_argument('--overlay',            type=int, default=0)
    p.add_argument('--delete_compressed',  type=str, default="yes")
    args, _ = p.parse_known_args()

    main(args)

