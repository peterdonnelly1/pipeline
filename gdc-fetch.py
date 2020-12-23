

# if need to bulk re-create the download flags:
#   find <DATA_DIRECTORY> -maxdepth 1 -type d -name "*" -exec mkdir {}_all_downloaded_ok \;
# eg:
#  find stad -maxdepth 1 -type d -name "*" -exec mkdir {}_all_downloaded_ok \;

import os
import re
import sys
import json
import gzip
import glob
import fnmatch
import requests
import random
import argparse
import tarfile
import pandas as pd
import shutil as sh
from   pathlib  import Path

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

FAIL       = 0
SUCCESS    = 1

OMIT       = -1
NOT_FOUND  = 0
FOUND      = 1

already_have_suffix = '_all_downloaded_ok' 

def main(args):
  
  DEBUG            = args.debug
  dataset          = args.dataset
  output_dir       = args.dataset
  base_dir         = args.base_dir
  uberlay          = args.uberlay
  overlay          = args.overlay
  case_filter      = args.case_filter 
  file_filter      = args.file_filter
  portal           = args.gdc_portal
  cleanup          = args.cleanup

	  
  if(os.path.isdir( output_dir )):
    user_input = input( "\033[1mWARNING: output directory \033[31;1;4m{:}\033[m\033[1m already exists, perhaps from a previous interrupted run. \
\n\033[31;1;4mf\033[m\033[1minish previous download or \
\n\033[31;1;4mp\033[m\033[1mromote all leaf files to their correct positions and delete all empty directories or \
\n\033[31;1;4mc\033[m\033[1mlean up unwanted files or \
\n\033[31;1;4mo\033[m\033[1mverlay new data (only consider cases we already have).\033[m or \
\n\033[31;1;4mu\033[m\033[1mberlay new data (consider cases we don't currently have)\033[m or \
\n\033[31;1;4md\033[m\033[1melete directory and start afresh?  \
\033[m".format(output_dir) )
  
    while True:
      if user_input=='f':
    	  break
      elif user_input=='d':
        try:
          sh.rmtree(output_dir)
        except OSError:
          pass
        os.makedirs( output_dir )
        break
      elif user_input=='u':
        uberlay="yes"
        break
      elif user_input=='o':
        overlay="yes"
        break
      elif user_input=='p':
        cleanup="yes"
        break
      else:
        print ("sorry, no such option" )
        exit(0)

  if DEBUG>0:
    print( f"\n{MIKADO}GDC-FETCH:     INFO:    technique for downloading a new TCGA cancer class ('project') where you want only cases that contain BOTH an image file and an rna_seq file:" )
    print( f"GDC-FETCH:     INFO:        1) define a case filter which requests all cases from the project which contain rna-seq files (see for example {CYAN}filters/TCGA-THYM_case_filter{RESET}{MIKADO})" )
    print( f"GDC-FETCH:     INFO:        2) define a file filter which requests all files which contain svs files                      (see for example {CYAN}filters/GLOBAL_file_filter_UQ{RESET}{MIKADO})" )
    print( f"GDC-FETCH:     INFO:     then run {CYAN}gdc-fetch{RESET}{MIKADO} two times, to get the rna_seq files and then the image files (order doesn't really matter but it's usually better to get the rna_seq files first because they're much smaller)" )
    print( f"GDC-FETCH:     INFO:     important: use {UNDER}{RED}f{RESET}{BOLD}inish{RESET}{MIKADO} option on the second run. (rhis option downloads additional files for cases that have already been downloaded)" )
    print( f'GDC-FETCH:     INFO:        3) {CYAN}./gdc-fetch.sh thym "filters/TCGA-THYM_case_filter" "filters/GLOBAL_file_filter_UQ"{RESET}{MIKADO}' )
    print( f'GDC-FETCH:     INFO:        4) {CYAN}./gdc-fetch.sh thym "filters/TCGA-THYM_case_filter" "filters/GLOBAL_file_filter_SVS"{RESET}{MIKADO}' )
    print( f"GDC-FETCH:     INFO:     explanation: because the file filter is applied to the already filtered cases, only cases which contain both svs and rna_seq files will be downloaded{RESET}\n" )

  else:
    pass

###########################################################################################################################################
# STEP 1: RETRIEVE CASE UUIDs OF CASES WHICH MEET SEARCH CRITERIA PROVIDED TO THE GDC API
###########################################################################################################################################

  xlay = "no"
  if ( uberlay=="yes" ) | ( overlay=="yes" ):
    xlay="yes"

  if cleanup=="yes":
    if DEBUG>0:
      print( f"GDC-FETCH:    about to promote all leaf files to their correct positions and delete all empty directories for output_dir = {MAGENTA}{output_dir}{RESET}" )
    
    result = promote_leaf_files    ( 0,  DEBUG,  output_dir, output_dir )   
    result = delete_unwanted_files ( 0,  DEBUG,  output_dir             )
    
    if DEBUG>0:
      print( "GDC-FETCH:    finished" )
    exit(0)

  if portal   == "main":
    cases_endpt = "https://api.gdc.cancer.gov/cases"
  elif portal == "legacy":
    cases_endpt = "https://api.gdc.cancer.gov/legacy/cases"
  else:
    print( "\nGDC-FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )


  cases_uuid_list = []

  if overlay=="no":

    if DEBUG>0:
      print( f"\nGDC-FETCH:  {BOLD} STEP 1:{RESET} about to retrieve case UUIDs of cases that meet the search criteria provided{RESET}" )
    
    fields = [
        "case_id"
        ]
    
    fields = ",".join(fields)
  
    with open( case_filter, 'r') as file:
      filters = json.load(file)
  
    params1 = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "JSON",
        "size": args.max_cases
        }
    
    try:
      response = requests.get( cases_endpt, params=params1 )
    except Exception as e:
      print ( f"{RED}GDC-FETCH:   FATAL:  no Internet connection?{RED}{RESET}" )
      print ( f"{RED}GDC-FETCH:   FATAL:  cannot continue - halting now{RESET}" )                 
      sys.exit(0)
      
    
    
    for case_entry in json.loads( response.content.decode("utf-8")) ["data"]["hits"]:
        cases_uuid_list.append(case_entry["case_id"])
    
    if DEBUG>1:
      print( "GDC-FETCH:  response (should be a json struct of the fields we requested. We are only interested in 'case_id') = {:}\033[m".format( response.text ) )

  else: # user selected 'overlay' mode. I.e. download additional files according to the filters, but only for cases we already have.
    walker = os.walk( output_dir, topdown=True )
    for root, dirs, files in walker:
      for d in dirs:
        fqd = root + '/' + d
        if DEBUG>0:
          print( "GDC-FETCH:        overlay mode: now examining: \033[1m{:}\033[m".format( fqd ) )
        if( os.path.isdir( fqd )):
          if fqd.endswith( already_have_suffix ):
            regex = r'.*\/(.*).*_all_downloaded_ok'
            matches = re.search( regex, fqd )
            case_uuid = matches.group(1)
            if DEBUG>0:
              print( "GDC-FETCH:        case from already_have_flag:      \033[1m{:}\033[m".format( case_uuid ) )          
            try:
              if DEBUG>0:
                print( "GDC-FETCH:        overlay mode and found case:      \033[1m{:}\033[m".format( case_uuid ) )
              cases_uuid_list.append(  case_uuid   )
            except:
              pass

  if DEBUG>0:
    print( f"GDC-FETCH:    INFO:   cases_uuid_list = {CYAN}{cases_uuid_list}{RESET}" )

  if DEBUG>0:
    print( f"GDC-FETCH:  {BOLD}STEP 2:{RESET} about to loop through each case UUID and request the UUIDs of associated files for each case\033[m" )

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
#            i.e. decompressed .gz filesis greater thansize, only some of the results will be returned.Thefromquery parameter specifies the first record to return out of the set of results. For example, if there are 20 cases returnedfrom thecasesendpoint, then settingfromto11will return results 12 to 20. Thefromparameter can be used in conjunctionwith thesizeparameter to return a specific subset of results.Example1curl'https://api.gdc.cancer.gov/files?fields=file_name&from=0&size=2&pretty=true'1import requests2import json34files_endpt ='https://api.gdc.cancer.gov/files'5pa
#            at this point we have all wanted files at the case id level
#      2f  set up and populate a new case_id subdirectory for each SVS file downloaded                - setup_and_fill_case_subdirs()
#            for each SVS file (n, SVS file)
#              make a new subdirectory at the case id level with the extension case_id-<n>             
#              copy the SVS file plus the RNA-SEQ file into the new subdirectory
#      2g  delete the original case_id directory                                                      - delete_unwanted_files()
#      2h  create a new case level subdirectory named to flag that the case was handled successfully  - _all_downloaded_ok()
#            checked on subsequent runs of gdc_fetch, so that files are not needlessly re-downloaded
#            especially SVS files, which can be extremely large (multi-gigabyte)
#  
###########################################################################################################################################
  
  n=0
  global_download_counter=0
  
  for case in cases_uuid_list:

    a = random.choice( range(150,230) )
    b = random.choice( range(200,235) )
    c = random.choice( range(150,230) )
    RAND="\033[38;2;{:};{:};{:}m".format( a,b,c )

    case_path = "{:}/{:}/".format( output_dir, case )
    
    if DEBUG>9:
      print( f"GDC-FETCH:        case_path = {MAGENTA}{case_path}{RESET}" )
  
    if ( ( uberlay=="yes" ) & ( os.path.isdir( case_path )==True  )  ) :
        print( f"GDC-FETCH:      {RAND}skipping {MAGENTA}{case_path}{RAND} and moving to next case{RESET}" )
    else:
      if DEBUG>0:
        print( f"GDC-FETCH:    INFO:   files downloaded so far {MIKADO}{global_download_counter}{RESET} (user defined max = {MIKADO}{args.global_max_downloads}{RESET})" )
  
      if  global_download_counter >=  args.global_max_downloads:
        if DEBUG>0:
          print( f"GDC-FETCH:    user defined maximumum number of files to be downloaded ({MIKADO}{RAND}{args.global_max_downloads}{RESET}) has been reached. Stopping." )
        break
      
      n+=1
  
      if DEBUG>0:
        print( f"GDC-FETCH:    INFO:   case {MIKADO}{n}{RESET} of {MIKADO}{len( cases_uuid_list)}{RESET}"  )
  
      if DEBUG>0:
        print( f"GDC-FETCH:    INFO:   case id {MIKADO}{case}{RESET}" )
  
      already_have_svs_file = False                                                                          # will be changed to True if an SVS file already exists & we are in uberlay mode
  
      already_have_flag = case_path[:-1] + already_have_suffix                                               # set on last download of this case, if there was one
  
      if DEBUG>99:
        print( "GDC-FETCH:      'already_have_flag'                           =  {:}{:}\033[m".format( RAND,  already_have_flag ) )
  
      if ( xlay=="no" ) & ( Path( already_have_flag ).is_dir()):
       # xlay=="yes" & already_have_flag     set  - files for this case were already successfully downloaded, and user is not asking us to fetch further files for the case, so skip and move to the next case
          print( f"GDC-FETCH:    INFO:   files already exist for case {RAND}{case}{RESET}" )
  
       # xlay=="no"  & already_have_flag not set  - download dir may or may not exist. Either: first ever download of this case, else user selected 'continue' or 'delete'
       # xlay=="yes" & already_have_flag not set  - download dir MUST exist (else xlay option wouldn't have been offered). User explicitly specificed xlay, but might also be first download of this case, or broken download or or else user selected 'xlay'
       # xlay=="yes" & already_have_flag     set  - download dir MUST exist (else xlay option wouldn't have been offered). User explicitly specificed xlay, so there should be NEW files to get. Normal scenario for 'xlay' option.
  
      if uberlay=="yes":
  
        if DEBUG>0:
          print ("GDC-FETCH:                                                       \033[1m{:}!!! uberlay mode\033[m".format ( RAND ) )     
  
        walker = os.walk( case_path )
        for root, _, files in walker:
          for f in files:
            if  ( ( fnmatch.fnmatch( f,"svs") )  ):                                                  # if we come across an svs file in the case folder     
              already_have_svs_file = True
              if DEBUG>0:
                print ("GDC-FETCH:                                                       \033[1m{:}already have an SVS file for this case ... skipping and moving to next case \033[m".format ( RAND ) )
  
      if already_have_svs_file == False:
  
        if overlay=="yes":
          if DEBUG>0:
            print ("GDC-FETCH:                                                       \033[1m{:}!!! overlay mode\033[m".format ( RAND ) )
            
        if DEBUG>1:
          print( "GDC-FETCH:    \033[1m2a:\033[m requesting file UUIDs for case                 {:}{:}\033[m".format( RAND, case )  )
          

  if DEBUG>0:
    print( "\nGDC-FETCH:    all done" )

#====================================================================================================================================================
# 2z VALIDATE CASE FILES
#
#  Ensure the case file is listed in the applicable TCGA master spreadsheet. If it's not, we can't use it, since it will have no truth values (cancer class/subtype)
#

def validate_case_file ( DEBUG, case ):

  cancer_class                           = args.dataset
  class_specific_global_data_location    = f"{args.base_dir}/{args.dataset}_global"
  class_specific_master_spreadsheet_name = f"{cancer_class}_mapping_file_MASTER"

  fqn = f"{class_specific_global_data_location}/{class_specific_master_spreadsheet_name}"
  if (DEBUG>0):
    print ( f"GDC-FETCH:    INFO:   about to open {CYAN}{cancer_class}{RESET} master spreadsheet to validate case:  {MAGENTA}{fqn}{RESET}")

  try:
    df = pd.read_csv( f"{fqn}", sep='\t' )
  except Exception as e:
    print ( f"{RED}GDC-FETCH:     FATAL: '{e}'{RESET}" )
    print ( f"{RED}GDC-FETCH:     FATAL:  explanation: the {CYAN}{cancer_class}{RESET}{RED} master mapping file ('{MAGENTA}{class_specific_master_spreadsheet_name}{RESET}{RED}') was not found in {MAGENTA}{class_specific_global_data_location}{RESET}{RED}{RESET}" )
    print ( f"{RED}GDC-FETCH:     FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)


  if DEBUG>9:  
    print ( f"GDC-FETCH:     INFO:    VALIDATING: about to check this case id: {case}" )

  start_row   = 2
  case_column = 1
  omit_column = 4
  
  outcome = NOT_FOUND                                                                                      # unless the case is located, outcome will remain at 'NOT_FOUND'
  for r in range(start_row, len(df)):
    c =  df.iloc[r, case_column]
    if DEBUG>99:
      print ( f"GDC-FETCH:     INFO:   {DIM_WHITE}checking case id {MIKADO}{case}{RESET} {DIM_WHITE} against spreadsheet case {CYAN}{c}{RESET}" )  
    if ( case==c ):
      omit_status =  df.iloc[r, omit_column]
      if  omit_status == OMIT:
        outcome = OMIT
        if DEBUG>0:
          print ( f"GDC-FETCH:    INFO:   {RED}directory (case) '{CYAN}{case}{RESET}'{RED} \r\033[85C is marked for omission. It will be skipped" )
        break
      else:
        outcome = FOUND
        if DEBUG>0:
          print ( f"GDC-FETCH:    INFO:   validation step: {GREEN}directory (case) '{CYAN}{case}{RESET}'{GREEN} \r\033[90C is present in the locally held copy of the TCGA {CYAN}{cancer_class}{RESET}{GREEN} master spreadsheet{RESET}" )
        break
    elif re.search( "_[0-9]", case ):
      if (case)[:-2] == c:
        omit_status =  df.iloc[r, omit_column]
        if  omit_status == OMIT:
          outcome = OMIT
          if DEBUG>0:
            print ( f"GDC-FETCH:    INFO:   {RED}case '{CYAN}{case}{RESET}'{RED} \r\033[85C is marked for omission. It will be skipped" )
          break
      else:
        outcome = FOUND
        if DEBUG>0:         
          print ( f"GDC-FETCH:     INFO:   {GREEN}parent of case '{CYAN}{case}{RESET}'{GREEN} \r\033[85C is listed the {CYAN}{cancer_class}{RESET}{GREEN} master spreadsheet{RESET}" )
        break
  else:
    print ( f"GDC-FETCH:    INFO:   {RED}validation step: directory (case) '{CYAN}{case}{RESET}'{RED} \r\033[90C is NOT present in the locally held copy of the TCGA {CYAN}{cancer_class}{RESET}{RED} master spreadsheet. This case and its files will not be downloaded.{RESET}" )


  return outcome

#====================================================================================================================================================
# 2a FETCH CASE FILE IDs

def fetch_case_file_ids( RAND, DEBUG, case, portal, file_filter, uberlay, overlay, already_have_flag ):

  if portal == "main":
    files_endpt = "https://api.gdc.cancer.gov/files"
  elif portal == "legacy":
    files_endpt = "https://api.gdc.cancer.gov/legacy/files"
  else:
    print( "\nGDC-FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )

  
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
  
  if DEBUG>99:
    print( "GDC-FETCH:          response (json list of file ids of hits)  =   {:}{:}\033[m".format(RAND, case_files.text ) )

  hits = json.loads( case_files.content.decode("utf-8"))["data"]["hits"]
  
  if DEBUG>9:
    print ( 'GDC-FETCH:          ["hits"]                                  =  \033[32;1m{:}\033[m'.format( hits )   )

  if not len(hits) == 0:                                                                                   # no files match the filter (which is perfectly possible)
    if DEBUG>9:
      print ( 'GDC-FETCH:          already_have_flag:                           \033[32;1m{:}\033[m'.format( already_have_flag )   )

    if ( overlay=="yes" ) | ( uberlay=="yes" ) :
      if Path( already_have_flag ).is_dir():	                                                           # uberlay or overlay mode and there are new files, so delete the already have flag to ensure integrity checking for this download	      
        os.rmdir ( already_have_flag )

      if DEBUG>0:
        print ( 'GDC-FETCH:    INFO:   new files to be downloaded so deleting:      \033[32;1m{:}\033[m'.format( already_have_flag )   )

    return SUCCESS, case_files

  else:
    if DEBUG>0:
      print ( 'GDC-FETCH:          no new files for this case - skipping' ) 
       
    return FAIL, case_files



#====================================================================================================================================================
# 2b DOWNLOAD CASE FILES

def download( RAND, DEBUG, output_dir, case_path, case, case_files, portal ):
    
  if DEBUG>1:
    print( "GDC-FETCH:    \033[1m2b:\033[m about to populate file UUID download list and request files" )
    
  file_uuid_list = []

  # (i) Populate the download list with the file_ids from the previous query
  for file_entry in json.loads(case_files.content.decode("utf-8"))["data"]["hits"]:

    file_uuid_list.append(file_entry["file_id"])
      
  if DEBUG>0:
    print( "GDC-FETCH:    INFO:   files to be downloaded for this case     =  {:}{:}\033[m".format( RAND, file_uuid_list) )
                                                                            # if the file in '.svs' (there can sometimes be one or more than one slide file)

# (ii) Request, download and save the files (there will only ever be ONE actual file downloded because the GDC portal will put multiple files into a tar archive)

  if portal == "main":
    data_endpt = "https://api.gdc.cancer.gov/data"
  elif portal == "legacy":
    data_endpt = "https://api.gdc.cancer.gov/legacy/data"  
  else:
    print( "\nGDC-FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )
  
  params = {"ids": file_uuid_list}
  
  response = requests.post( data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})
  
  response_head_cd = response.headers["Content-Disposition"]

  if DEBUG>9:
    print( "GDC-FETCH:          response.headers[Content-Type             = {:}'{:}'\033[m".format( RAND, response_head_cd ) )
  
  download_file_name = re.findall("filename=(.+)", response_head_cd)[0]                                    # extract filename from HTTP response header
 
  if DEBUG>9:
    print( "GDC-FETCH:          response.headers[Content-Disposition]     = {:}'{:}'\033[m".format( RAND, download_file_name ) )
    print( "GDC-FETCH:          download_file_subdir_name                 = {:}'{:}'\033[m".format( RAND, case_path ) )

  if not os.path.isdir ( case_path ):
    os.makedirs( case_path )
  
  download_file_fq_name = "{:}/{:}".format( case_path, download_file_name )

  if DEBUG>9:
    print( "GDC-FETCH:          download_file_fq_name                     = {:}'{:}'\033[m".format( RAND, download_file_fq_name ) )

  with open(download_file_fq_name, "wb") as output_file_handle:                                            # save the downloaded file
    output_file_handle.write(response.content)

  # if it's not already a tarball we will turn it into one to allow for uniform processing
  if not download_file_fq_name.endswith("tar.gz"):                                                         
    
    if DEBUG>9:
      print( "GDC-FETCH:            SINGLETON: will create tarball from:      {:}'{:}'\033[m".format( RAND, download_file_fq_name ) )

    standard_name   = "STANDARD_NAME.tar" 
    tarfile_fq_name = "{:}{:}".format( case_path,  standard_name      ) 
    arcpath         = f"arcpath/{download_file_name}"

    if DEBUG>9:
      print( "GDC-FETCH:            SINGLETON: arcpath:                       {:}'{:}'\033[m".format( RAND, arcpath ) )

    try:
      tf = tarfile.open( tarfile_fq_name, 'x:gz' )                                                         # create new tarfile called STANDARD_NAME.tar in case_path
    except Exception:
      pass

    try:
      tf.add( download_file_fq_name, arcname=arcpath )
    except Exception:
      pass
 
    if DEBUG>9:
      print( "GDC-FETCH:            SINGLETON: tarball created ok:            {:}'{:}'\033[m".format( RAND, standard_name ) )
        
    try:
      tf.close()
    except Exception:
      pass

    try:
      os.remove( download_file_fq_name )
      if DEBUG>9:
        print( "GDC-FETCH:            SINGLETON: now deleting:                  {:}'{:}'\033[m".format( RAND, tarfile_fq_name ) )
    except Exception:
      pass

    return standard_name                                                                                   # the case where there was no tarball, so we created one
  
  return download_file_name                                                                                # the case where a tarball was downloaded


#====================================================================================================================================================
# 2c UNPACK TARBALL
 
def unpack_tarball ( RAND, DEBUG, case_path, tarfile_name ):

  tarfile_fq_name = "{:}/{:}".format( case_path, tarfile_name )   
  tar = tarfile.open( tarfile_fq_name )

  try:
    tar = tar.extractall( path=case_path )
  except Exception:
    pass

  try:
    tar.close()
  except Exception:
    pass

  if DEBUG>9:
    print( "GDC-FETCH:          tarball unpacked ok\033[m" )

  #try:
  #  os.remove( tarfile_fq_name )
   # if DEBUG>0:
    #  print( "GDC-FETCH:    \033[1m2c:\033[m now deleting tarball" )
  #except Exception:
   # pass

  return SUCCESS

#====================================================================================================================================================
# 2d DECOMPRESS ANY GZ FILES WHICH MAY HAVE BEEN DOWNLOADED

def decompress_gz_files( RAND, DEBUG, case_path ):

  if DEBUG>1:
    print( "GDC-FETCH:    \033[1m2d:\033[m unzipping all gz files in case path           {:}'{:}'\033[m, using match pattern {:}'{:}*.gz'\033[m".format( RAND, case_path, RAND, case_path ) )
    
  for dir_path, _, files in os.walk( case_path ):
    
    for gz_candidate in files:
      
      if  ( ( fnmatch.fnmatch( gz_candidate, "*.gz") )  ):                                                  # if it's a gz file

        if DEBUG>9:
          print( "GDC-FETCH:             opening                                  {:}'{:}'\033[m".format( RAND, gz_candidate ) )
    
        fq_name = f"{dir_path}/{gz_candidate}"

        with gzip.open( fq_name, 'rb') as f:
          s = f.read()
            
        output_name    = fq_name[:-3]                                                                      # remove '.gz' extension from the filename
      
        if DEBUG>9:
          print( "GDC-FETCH:             saving decompressed file as              {:}'{:}'\033[m".format( RAND, output_name ) )

        with open(output_name, 'wb') as f:                                                                 # store uncompressed data
          f.write(s)
        f.close
     
        if DEBUG>9:                                                                                        # delete the gz file
          print ( "GDC-FETCH:          about to delete                             {:}'{:}'\033[m".format( RAND, fq_name )      )
        try:
          os.remove( fq_name )
        except Exception:
          pass
  return SUCCESS
  
#====================================================================================================================================================
# 2e PROMOTE LEAF FILES UP INTO CASE DIRECTORY


def promote_leaf_files( RAND, DEBUG, output_dir, case_path  ):

  if DEBUG>1:
    print( f"GDC-FETCH:    {WHITE}2e:{RESET} about to promote leaf files of interest up into the case directory{RESET}" ) 

  for run in range (0,4):                                                                                   # assuming that files are nested at most 3 deep

    if DEBUG>9:
      print( f"GDC-FETCH:            promote(): run                        = {CYAN}{run}{RESET}" ) 
          
    for dir_path, dirs, files in os.walk( case_path, topdown=True ):
     
      if  dir_path == output_dir:                                                                           # ignore the first level directory altogether. They're shouldn't be any files in this directory
        break
        
      for f in files:
  
        # if <the directory TWO above> is <the output directory> then break (to prevent over-promoting)
        file_path=Path( f"{dir_path}/{f}" )
        output_dir=Path(output_dir)
        grandparent_directory = file_path.parents[1]                                                       # Don't promote file if it the grandparent directory is the root directory
        if DEBUG>9:
          print( f"GDC-FETCH:            promote(): file_path                        = {MAGENTA}{file_path}{RESET}"             )           
          print( f"GDC-FETCH:            promote(): dir_path                         = {MAGENTA}{dir_path}{RESET}"              )           
          print( f"GDC-FETCH:            promote(): grandparent directory            = {MAGENTA}{grandparent_directory}{RESET}" )
          print( f"GDC-FETCH:            promote(): output_dir                       = {MAGENTA}{output_dir}{RESET}"            ) 
        if grandparent_directory==output_dir:
          if DEBUG>9:
            print( f"{ORANGE}GDC-FETCH:            promote(): grandparent_directory is the output_dir {MAGENTA}('{output_dir}'){ORANGE}, so we will not promote this file{RESET}" )  
          break
        
        fq_name = "{:}{:}".format( case_path, f )
  
        if DEBUG>9:
          print ( f"GDC-FETCH:            promote(): case_path                        = {RAND}'{case_path}'{RESET}" )
          print ( f"GDC-FETCH:            promote(): f                                = {RAND}'{f}'{RESET}"         )
                                                                            
        if  ( ( fnmatch.fnmatch( f, "MANIFEST.*" ) )  ):                                                   # have to delete, because "MANIFEST.TXT" always only one level down, not two levels down; and there are text files that we DON'T want to delete, so we can't pick up later in the general clean up function
          if DEBUG>9:
            print ( "GDC-FETCH:            promote(): about to delete                  = {:}'{:}'\033[m".format( RAND, fq_name )      )
          try:
            os.remove( fq_name )
          except Exception:
            pass
          
        elif ( fq_name.endswith("*.tar") ) | ( fq_name.endswith("*.gz") ):                                 # these will be deleted later anyway, and we don't want them promoted up above output_dir
          pass
  
        else: 
          fq_name      = "{:}/{:}".format( dir_path, f )
          if DEBUG>9:
            print ( f"GDC-FETCH:            promote(): moving file up a level: filename = {RAND}'{fq_name}'{RESET}" )
          move_to_name = "{:}/../{:}".format( dir_path, f )
          if DEBUG>9:
            print ( f"GDC-FETCH:            promote(): moving file up a level: new name = {RAND}'{move_to_name}'{RESET}" )     
    
          sh.move( fq_name, move_to_name )

  return SUCCESS
 
#====================================================================================================================================================
# 2f SET UP AND POPULATE A NEW CASE_ID SUBDIRECTORY FOR EACH SVS FILE

#  For each SVS file (n, SVS file)
#   make a new subdirectory at the case id level with the name case_id_<n>             
#   copy the SVS file plus the RNA-Seq file into the new subdirectory
#
#  Caters for the situation where a case has more than one slide by creating a new 'pseudo-case' for each slide.
#  In this fashion:
#     Cases with a  single   slide file  and no       rna-seq file are handled
#     Cases with a  multiple slide files and no       rna-seq file are handled
#     Cases with no          slide files and a single rna-seq file are handles
#  However:
#     Cases with a  multiple slide files and multiple rna-seq file are NOT handled
#     For this scenario, the first rna-seq file the code comes across will be used and the others ignored
#
#  The effort required to cater for the last scenario didn't seem to be justified, as there seem to be very few such examples (whereas it's not uncommon for there to be more than one slide)
#  If it becomes necessary to cater for the last scenario, just create doubly extended case/directory names, like this:
#
#     e4344668-5a50-4dde-8eec-f4d7f01f99fd_3_2thym filters/TCGA-THYM_case_filter filters/GLOBAL_file_filter_SVS 
#
#  where the first suffix refers to the slide file used (as is the case now) and the second suffix would refer to the rna_seq file used
#  note that the suffix numbers are only used to ensure each slide/rna_seq combination (i) will have a unique directory name (ii) from which its easy to recover the parent case/directory
#  

def setup_and_fill_case_subdirs    ( RAND, DEBUG, case_path ):

  if DEBUG>1:
    print( "GDC-FETCH:    \033[1m2f:\033[m about to set up and populate a new case_id subdirectory for each svs file found" )
    
    svs_count   = 0
    other_count = 0

    for f in os.listdir( case_path ):                                                                      # case path is the parent case/dir name for samples that derive from this case (one additional dir for each slide in the case path)
      if f.endswith(".svs"):                                                                               # if the file in '.svs' (there can sometimes be one or more than one slide file)
        svs_count+=1                                                                                       
        
        new_dir_name = case_path[:-1] + '_' + str( svs_count )                                             # make a new child directory and name accordingly
        if DEBUG>0:
          print( "GDC-FETCH:          about to create new directory             '{:}{:}'\033[m".format        ( RAND, new_dir_name           ) )
        try:
          new_dir = os.mkdir( new_dir_name )
        except OSError:
          pass

        if DEBUG>0:
          print( f"GDC-FETCH:            SVS file count = {RAND}{svs_count} and file name is     '{RAND}{f}'{RESET}"  )        
          print( f"GDC-FETCH:            about to move SVS file to new directory '{RAND}{new_dir_name}{RESET}' "      )
        existing_SVS_FQ_name = str(   case_path  )       +           str(f)                                # FQ name of the slide we are about to move
        if DEBUG>0:
          print( f"GDC-FETCH:            old FQ name =                           '{RAND}{existing_SVS_FQ_name}{RESET}'" )   
        new_SVS_FQ_name      = str( new_dir_name ) + '/' +           str(f)                                # thym filters/TCGA-THYM_case_filter filters/GLOBAL_file_filter_SVS FQ name of the destination. ('new_SVS_FQ_name' has the _<n> extension)
        if DEBUG>0:
          print( f"GDC-FETCH:            new FQ name =                           '{RAND}{new_SVS_FQ_name}{RESET}'" )
        os.rename(   existing_SVS_FQ_name, new_SVS_FQ_name     )                                           # move this slide into the newly created directory (reducing the number of slides in the case_path by one)                          

    # At this point we created a (suffix named) child directory for each additional slide which may exist in case_path, and move one of the additional slides into each
    # Next we will move a copy of rna_file(s) (if there are any) from the case_path into each of new (suffix named) child directories
    
    for f in os.listdir( case_path ):
      if f.endswith(".txt"):
        other_count+=1
        
        for n in range( 0, svs_count ) :                                                                   # copy it into each of the 'svs_count' new folders created just above
        
          target_dir_name = case_path[:-1] + '_' + str( n+1 )
          
          if DEBUG>0:
            print( "GDC-FETCH:          n = {:}{:}\033[m of {:}{:}\033[m".format                              ( RAND, n, RAND, svs_count     ) )
            print( "GDC-FETCH:          OTHER file count = {:}{:} and file name is     '{:}{:}'\033[m".format ( RAND, other_count, RAND, f   ) )
            print( "GDC-FETCH:            about to copy file to new directory     '{:}{:}'\033[m".format      ( RAND, target_dir_name        ) )
          if f.endswith(".txt"):
            existing_OTHER_FQ_name = str( case_path )       +        str(f)                                # FQ name of the file we are about to move
          if DEBUG>0:
            print( "GDC-FETCH:            old FQ name =                           '{:}{:}'\033[m".format      ( RAND, existing_OTHER_FQ_name ) )
          new_other_FQ_name        = str( target_dir_name ) + '/' +  str(f)                                # FQ name of the destination
          if DEBUG>0:
            print( "GDC-FETCH:            new FQ name =                           '{:}{:}'\033[m".format      ( RAND, new_other_FQ_name      ) )		  
          sh.copyfile( existing_OTHER_FQ_name, new_other_FQ_name )                                         # move this rna_seq file to the target directory (reducing the number of rna_seq files in case_path by one)                       

      else:
        if DEBUG>9: 
          print( "GDC-FETCH:          this file will be ignored                 '{:}{:}'\033[m".format       ( RAND, f                     ) )
      

#====================================================================================================================================================
# 2g DELETE UNWANTED FILES AND EMPTY DIRECTORIES

def delete_unwanted_files( RAND, DEBUG, output_dir ):
	
  if DEBUG>1:
    print( "GDC-FETCH:    \033[1m2g:\033[m about to delete temp files and directories" )
    

  if DEBUG>9:
    print( f"GDC-FETCH:          root directory for deleting is:              {MAGENTA}{output_dir}{RESET}" )

  walker = os.walk( output_dir, topdown=False )

  for root, dirs, files in walker:

    for f in files:
      fqf = root + '/' + f
      if DEBUG>99:
        print( "GDC-FETCH:          examining file:                              {:}{:}\033[m".format( RAND, fqf ) )
      if ( f.endswith("tar") ) | ( f.endswith("gz") ): 
        try:
          if DEBUG>99:
            print( "GDC-FETCH:              will delete                              {:}{:}\033[m".format( RAND, fqf ) )
          os.remove( fqf )
        except:
          pass

    for d in dirs:
      fqd = root + '/' + d
      if DEBUG>99:
        print( "GDC-FETCH:          examining directory:                         {:}{:}\033[m".format( RAND, fqd ) )
      if( os.path.isdir( fqd )):
        if not d.endswith( already_have_suffix ): 
          try:
            if DEBUG>99:
              print( "GDC-FETCH:          will delete if empty:                        {:}{:}\033[m".format( RAND, fqd ) )
            os.rmdir( fqd )
          except:
            pass

    
  return SUCCESS

#====================================================================================================================================================
#  2h  create a new case level subdirectory named to indicate the case was handled successfully

def _all_downloaded_ok( RAND, DEBUG, case_path ):

  if DEBUG>1:
    print( "GDC-FETCH:    \033[1m2h:\033[m about to create new case level subdirectory, named to indicate the case was handled successfully" )
    
  try:
    fqn = case_path[:-1] + already_have_suffix
    os.mkdir( fqn )                   
  except Exception as e:
    print ( f"GDC-FETCH:    {ORANGE}INFO:   download flag {CYAN}{fqn}{RESET}{ORANGE} already exists{RESET}")


  return SUCCESS
  
#====================================================================================================================================================
      
if __name__ == '__main__':
	
    p = argparse.ArgumentParser()

    p.add_argument('--debug',                type=int, default=1)
    p.add_argument('--dataset',              type=str,                                          required=True      )
    p.add_argument('--base_dir',             type=str,                                          required=True      )
    p.add_argument('--gdc_portal',           type=str, default="main")
    p.add_argument('--case_filter',          type=str, default="dlbc_case_filter")
    p.add_argument('--file_filter',          type=str, default="dlbc_file_filter_just_rna-seq")
    p.add_argument('--max_cases',            type=int, default=5)
    p.add_argument('--max_files',            type=int, default=10)
    p.add_argument('--global_max_downloads', type=int, default=200)
    p.add_argument('--uberlay',              type=str, default="no")
    p.add_argument('--overlay',              type=str, default="no")
    p.add_argument('--delete_compressed',    type=str, default="yes")
    p.add_argument('--cleanup',              type=str, default="no")

    args, _ = p.parse_known_args()

    main(args)


# try using bash: 
#   
# first issue IFS=$'\n' 
#
# then either:
#
#   for x in * ; do mv "$x"/*/* "$x"/ ; done   << didn't try this version
#
# or

# BE VERY CAREFUL WHEN USING THE FOLLOWING COMMAND. YOU MUST CHANGE TO THE WORKING DIRECTORY FIRST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#    for x in stad/* ; do [ -d $x ] && ( cd $x ; pwd ; mv * .. ; cd ../.. ) ; done    << this is the version I used and it worked. Had to apply it two times
#    for x in stad/* ; do [ -d $x ] && ( cd $x ; pwd ; mv stad/*.txt .. ; cd ../.. ) ; done
#
#  RELATED:
# 
# to count number of empty directories/files:
#
# find /path/ -empty -type d | wc -l
# find /path/ -empty -type f | wc -l
#
# to count empty directories/files:
#
#  find . -empty -type d -delete
#  find . -empty -type f -delete
#
