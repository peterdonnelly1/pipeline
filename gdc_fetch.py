import os
import re
import json
import gzip
import glob
import shutil as sh
import fnmatch
import requests
import random
import argparse
import tarfile
from   pathlib  import Path

SUCCESS = 1

def main(args):
  
  DEBUG        = args.debug
  output_dir   = args.output_dir
  disease_type = args.disease_type
  


  while True:
	  
    if(os.path.isdir(output_dir)):
      user_input = input( "\033[1mWARNING: directory named \033[31;1;4m{:}\033[m\033[1m already exists, perhaps from previous interrupted run. \033[31;1;4mc\033[m\033[1momplete previous download or \033[31;1;4md\033[m\033[1melete directory and start afresh?".format(output_dir) )

    if user_input=='c':
  	  break
    elif user_input=='d':
      try:
        sh.rmtree(output_dir)
      except OSError:
        pass
      os.makedirs(output_dir)
      break
    else:
      print ("sorry, that's not an available option" )



###########################################################################################################################################
# STEP 1: RETRIEVE CASE UUIDs OF CASES WHICH MEET SEARCH CRITERIA PROVIDED TO API
###########################################################################################################################################

  cases_endpt = "https://api.gdc.cancer.gov/cases"

  if DEBUG>0:
    print( "GDC_DOWNLOAD:  \033[1mSTEP 1:\033[m about to retrieve case UUIDs of cases which meet the provided search criteria" )
    print( "GDC_DOWNLOAD:  disease_type = \033[36;1m{:}\033[m".format( disease_type) )
  
  fields = [
      "case_id"
      ]
  
  fields = ",".join(fields)
  
  # This set of filters is nested under an 'and' operator.
  filters = {
      "op": "and",
      "content":[
          {
          "op": "in",
          "content":{
              "field": "cases.disease_type", 
              "value": disease_type
              }
          },
          {
          "op": "in",
          "content":{
              "field": "cases.project.project_id", 
              "value": ["TCGA-UVM"]
              }
          },
          {
          "op": "in",
          "content":{
              "field": "files.data_type", 
              "value": ["Gene Expression Quantification", "Slide Image" ] 
              }
          },
          {
          "op": "in",
          "content":{
              "field": "files.experimental_strategy",
              "value": ["Tissue Slide", "RNA-Seq"]
              }
          }
      ]
  }
  
  # With a GET request, the filters parameter needs to be converted from a dictionary to JSON-formatted string
  params1 = {
      "filters": json.dumps(filters),
      "fields": fields,
      "format": "JSON",
      "size": args.max_cases
      }
  
  response = requests.get(cases_endpt, params = params1)
  
  cases_uuid_list = []
  
  for case_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
      cases_uuid_list.append(case_entry["case_id"])
  
  if DEBUG>0:
    print( "GDC_DOWNLOAD:  response (should be a json struct of the fields we requested. We are only interested in 'case_id') = {:}\033[m".format( response.text ) )
    print( "GDC_DOWNLOAD:  cases_uuid_list = \033[36;1m{:}\033[m".format( cases_uuid_list) )


  if DEBUG>0:
    print( "GDC_DOWNLOAD:  \033[1mSTEP 2:about to loop through each case UUID and request the UUIDs of associated files for each case\033[m" )


###########################################################################################################################################
# STEP 2: LOOP THROUGH EACH CASE ID IN cases_uuid_list AND PROCESS
# 
# Pseudocode:
#    for each case:
#      2a  fetch file ids of files of interest
#      2b  download its files
#      2c  unpack the tarball just download
#      2d  decompress any gz files created upon decompression the tarball
#      2e  delete_unwanted_files
#      2f  place a flag in case_id subdir to indicate that the case was handled ok
#  
###########################################################################################################################################
  
  n=0
  
  for case in cases_uuid_list:
    
    a = random.choice( range(128,230) )
    b = random.choice( range(200,230) )
    c = random.choice( range(128,230) )
    RC="\033[38;2;{:};{:};{:}m".format( a,b,c )
  
    n+=1
    
    case_path = "{:}/{:}/".format( output_dir, case )

    if DEBUG>0:
      print( "\nGDC_DOWNLOAD:    case {:}{:}\033[m of {:}{:}\033[m".format( RC, n, RC, len(cases_uuid_list) ) )

    if Path( case_path +  '/files_downloaded_ok.flag').is_file():                                          # Id the files for this case were already previously downloaded, then move to next case
        print( "GDC_DOWNLOAD:    \033[1mfiles already exist for case = {:}{:} ... skipping and moving to next case\033[m".format( RC, case ) )

    else:
      if DEBUG>0:
        print( "GDC_DOWNLOAD:    \033[mrequesting file UUIDs for with case {:}{:}\033[m".format( RC, case ) )

      case_files = fetch_case_file_ids ( RC, DEBUG, case )
      
      IS_TAR_ARCHIVE, tarfile_name = download_and_save_case_files ( RC, DEBUG, case_path, case_files  )
      
      if IS_TAR_ARCHIVE:
        result = unpack_tarball       ( RC, DEBUG, case_path, tarfile_name   )
    
      result = decompress_gz_files    ( RC, DEBUG, case_path                 )
      result = delete_unwanted_files  ( RC, DEBUG, case_path                 )
      result = place_result_flag      ( RC, DEBUG, case_path                 )

    if DEBUG>0:
      print( "\nGDC_DOWNLOAD:    all done".format )

#====================================================================================================================================================
# 2a FETCH FILE IDs

def fetch_case_file_ids( RC, DEBUG, case ):

  files_endpt = "https://api.gdc.cancer.gov/files"
  
  # This set of filters is nested under an 'and' operator.
  filters = {
      "op": "and",
      "content":[
          {
          "op": "in",
          "content":{
              "field": "cases.case_id",
              "value": case
              }
          },
        {
        "op": "in",
        "content":{
            "field": "files.data_type", 
            "value": ["Gene Expression Quantification"] 
            }
        },
        {
        "op": "exclude",
        "content":{
            "field": "files.file_name",
            "value": ["*FPKM.txt.gz"]
            }
        },
        {
        "op": "exclude",
        "content":{
            "field": "files.file_name",
            "value": ["*counts.gz"]
            }
        },
        {
        "op": "in",
        "content":{
            "field": "files.experimental_strategy",
            "value": ["RNA-Seq"]
            }
        }
      ]
  }
  
  # Here a GET is used, so the filter parameters should be passed as a JSON string.
  
  params2 = {
      "filters": json.dumps(filters),
      "fields": "file_id",
      "format": "JSON",
      "size": args.max_files
      }

  case_files = requests.get(files_endpt, params=params2)
  
  if DEBUG>1:
    print( "GDC_DOWNLOAD:      response (should be a json struct of hits, including the file uuids of hits) = {:}{:}\033[m".format(RC, case_files.text ) )
  
  return case_files

#====================================================================================================================================================
# 2b DOWNLOAD CASE FILES

def download_and_save_case_files( RC, DEBUG, case_path, case_files ):
    
  if DEBUG>0:
    print( "GDC_DOWNLOAD:    \033[1m2b:\033[m About to populate file UUID download list and request files" )
    
  file_uuid_list = []

  # (i) Populate the download list with the file_ids from the previous query
  for file_entry in json.loads(case_files.content.decode("utf-8"))["data"]["hits"]:

    file_uuid_list.append(file_entry["file_id"])
      
  if len(file_uuid_list)>1 :
    IS_TAR_ARCHIVE=True
  else:
    IS_TAR_ARCHIVE=False
      
  if DEBUG>0:
    print( "GDC_DOWNLOAD:      file_uuid_list (should be a list of just file uuids) = {:}{:}\033[m".format( RC, file_uuid_list) )


  # (ii) Request, download and save the files (there will only ever be ONE actual file downloded because the GDC portal will put multiple files into a tar archive)
  data_endpt = "https://api.gdc.cancer.gov/data"
  
  params = {"ids": file_uuid_list}
  
  response = requests.post( data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})
  
  response_head_cd = response.headers["Content-Disposition"]

  if DEBUG>1:
    print( "GDC_DOWNLOAD:      response.headers[Content-Disposition] = {:}'{:}'\033[m".format( RC, response_head_cd ) )
  
  download_file_name = re.findall("filename=(.+)", response_head_cd)[0]                                            # extract filename from HTTP response header
 
  if DEBUG>1:
    print( "GDC_DOWNLOAD:      name of downloaded repository extracted from 'response.headers[Content-Disposition]' = {:}{:}'\033[m".format( RC, download_file_name ) )
    print( "GDC_DOWNLOAD:      download_file_subdir_name = {:}'{:}'\033[m".format( RC, case_path ) )

  os.makedirs( case_path )
      
  download_file_fq_name = "{:}/{:}".format( case_path, download_file_name )

  if DEBUG>1:
    print( "GDC_DOWNLOAD:      download_file_fq_name = {:}'{:}'\033[m".format( RC, download_file_fq_name ) )

  with open(download_file_fq_name, "wb") as output_file_handle:
      output_file_handle.write(response	.content)

  return IS_TAR_ARCHIVE, download_file_name
  
#====================================================================================================================================================
# 2c UNPACK TARBALL
 
def unpack_tarball ( RC, DEBUG, case_path, tarfile_name ):
  
  if DEBUG>0:
    print( "GDC_DOWNLOAD:      \033[1m2c:\033[m about to unpack tarball {:}{:}'\033[m to {:}'{:}'\033[m".format( RC, tarfile_name, RC, case_path ) )

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
    print( "GDC_DOWNLOAD:        tarball unpacked Ok\033[m")

  return SUCCESS
 
#====================================================================================================================================================
# 2d DECOMPRESS ANY GZ FILES WHICH MAY HAVE BEEN DOWNLOADED

def decompress_gz_files( RC, DEBUG, case_path ):

  for gz_file in glob.glob( "{:}/*.gz".format( case_path ) ):

    with gzip.open( gz_file, 'rb') as f:
      s = f.read()
        
    # Now store the uncompressed data
    output_name    = gz_file[:-3]                                                                      # remove the '.gz' from the filename
  
    # store uncompressed file data from 's' variable
    with open(output_name, 'wb') as f:
      f.write(s)
    f.close

  return SUCCESS
  
  
  
#====================================================================================================================================================
# 2e DELETE UNWANTED FILES

def delete_unwanted_files( RC, DEBUG, case_path ):
	
  # delete unwanted files
  walker = os.walk(case_path)
  for case_path, _, files in walker:
    for f in files:
      if  ( ( fnmatch.fnmatch(f,"*.gz") )  | ( fnmatch.fnmatch(f,"*.tar")  )  | ( fnmatch.fnmatch(f,"MANIFEST.*")  ) ):
        fq_name="{:}/{:}".format( case_path, f ) 
        if DEBUG>1:
          print ( "GDC_DOWNLOAD:      delete unwanted file: {:}".format(fq_name) )
        os.remove( fq_name )

  return SUCCESS
  
#====================================================================================================================================================
#  2f  place a flag in the case_id subdir to indicate that the case was handled ok

def place_result_flag( RC, DEBUG, case_path ):

  Path( case_path +  '/files_downloaded_ok.flag').touch()

  return SUCCESS
  
#====================================================================================================================================================
      
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--debug',              type=int, default=0)
    p.add_argument('--disease_type',       type=str, default="glioma")  
    p.add_argument('--max_cases',          type=int, default=5)
    p.add_argument('--max_files',          type=int, default=10)
    p.add_argument('--output_dir',         type=str, default="output")
    p.add_argument('--delete_compressed',  type=str, default="yes")    
    args, _ = p.parse_known_args()

    main(args)

