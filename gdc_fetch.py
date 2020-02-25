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

FAIL    = 0
SUCCESS = 1

already_have_suffix = '_all_downloaded_ok' 

def main(args):
  
  DEBUG            = args.debug
  output_dir       = args.output_dir
  uberlay          = args.uberlay
  overlay          = args.overlay
  case_filter      = args.case_filter 
  file_filter      = args.file_filter
  portal           = args.gdc_portal
  cleanup          = args.cleanup

	  
  if(os.path.isdir( output_dir )):
    user_input = input( "\033[1mWARNING: directory named \033[31;1;4m{:}\033[m\033[1m already exists, perhaps from previous interrupted run. \
\033[31;1;4mf\033[m\033[1minish previous download or \
\033[31;1;4mc\033[m\033[1mlean up unwanted files or \
\033[31;1;4mo\033[m\033[1mverlay new data\033[m or \
\033[31;1;4mu\033[m\033[1mberlay new data\033[m or \
\033[31;1;4md\033[m\033[1melete directory and start afresh?  \
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
      elif user_input=='c':
        cleanup="yes"
        break
      else:
        print ("sorry, no such option" )
        exit(0)

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
      print( "GDC_FETCH:    about to cleanup out \033[1m{:}\033[m".format( output_dir ) )
    
    result = delete_unwanted_files( 0, DEBUG, output_dir, 0 )
    
    if DEBUG>0:
      print( "GDC_FETCH:    finished" )
    exit(0)

  if portal   == "main":
    cases_endpt = "https://api.gdc.cancer.gov/cases"
  elif portal == "legacy":
    cases_endpt = "https://api.gdc.cancer.gov/legacy/cases"
  else:
    print( "\nGDC_FETCH:  \033[1mNo GDC endpoint corresponds to that URL\033[m " )


  cases_uuid_list = []

  if overlay=="no":

    if DEBUG>0:
      print( "\nGDC_FETCH:  \033[1mSTEP 1:\033[m about to retrieve case UUIDs of cases that meet the search criteria provided" )
    
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
    
    response = requests.get(cases_endpt, params=params1)
    
    for case_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
        cases_uuid_list.append(case_entry["case_id"])
    
    if DEBUG>0:
      print( "GDC_FETCH:  response (should be a json struct of the fields we requested. We are only interested in 'case_id') = {:}\033[m".format( response.text ) )

  else: # user selected 'overlay' mode. I.e. download additional files according to the filters, but only for cases we already have.
    walker = os.walk( output_dir, topdown=True )
    for root, dirs, files in walker:
      for d in dirs:
        fqd = root + '/' + d
        if DEBUG>0:
          print( "GDC_FETCH:        overlay mode: now examining: \033[1m{:}\033[m".format( fqd ) )
        if( os.path.isdir( fqd )):
          if fqd.endswith( already_have_suffix ):
            #regex = r'.*\/([.]*){:}'.format( already_have_suffix )
            regex = r'.*\/(.*).*_all_downloaded_ok'
            matches = re.search( regex, fqd )
            case_uuid = matches.group(1)
            if DEBUG>0:
              print( "GDC_FETCH:        case from already_have_flag:      \033[1m{:}\033[m".format( case_uuid ) )          
            try:
              if DEBUG>0:
                print( "GDC_FETCH:        overlay mode and found case:      \033[1m{:}\033[m".format( case_uuid ) )
              cases_uuid_list.append(  case_uuid   )
            except:
              pass

  if DEBUG>0:
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
    RC="\033[38;2;{:};{:};{:}m".format( a,b,c )

    if DEBUG>0:
      print( "\nGDC_FETCH:    downloads so far \033[1m{:}{:} ({:}{:}\033[m user defined max downloads)".format( RC, global_download_counter, RC, args.global_max_downloads ) )

    if  global_download_counter >  args.global_max_downloads:
      if DEBUG>0:
        print( "GDC_FETCH:    user defined maximumum number of downloads (\033[1m{:}{:}\033[m) has been reached. Stopping.".format( RC, args.global_max_downloads ) )
      break
    
    n+=1
    
    case_path = "{:}/{:}/".format( output_dir, case )
       
    if DEBUG>0:
      print( "GDC_FETCH:    case {:}{:}\033[m of {:}{:}\033[m".format( RC, n, RC, len( cases_uuid_list) ) )

    if DEBUG>0:
      print( "GDC_FETCH:    case id \033[1m{:}{:}\033[m".format( RC, case ) )

    already_have_svs_file = False                                                                          # will be changed to True if an SVS file already exists & we are in uberlay mode

    already_have_flag = case_path[:-1] + already_have_suffix                                               # set on last download of this case, if there was one

    if DEBUG>99:
      print( "GDC_FETCH:      'already_have_flag'                           =  {:}{:}\033[m".format( RC,  already_have_flag ) )

    if ( xlay=="no" ) & ( Path( already_have_flag ).is_dir()):
     # xlay=="yes" & already_have_flag     set  - files for this case were already successfully downloaded, and user is not asking us to fetch further files for the case, so skip and move to the next case
        print( "GDC_FETCH:   \033[1m files already exist for case                   =        {:}{:} \033[m                    ... skipping and moving to next case\033[m".format( RC, case ) )

     # xlay=="no"  & already_have_flag not set  - download dir may or may not exist. Either: first ever download of this case, else user selected 'continue' or 'delete'
     # xlay=="yes" & already_have_flag not set  - download dir MUST exist (else xlay option wouldn't have been offered). User explicitly specificed xlay, but might also be first download of this case, or broken download or or else user selected 'xlay'
     # xlay=="yes" & already_have_flag     set  - download dir MUST exist (else xlay option wouldn't have been offered). User explicitly specificed xlay, so there should be NEW files to get. Normal scenario for 'xlay' option.

    if uberlay=="yes":

      if DEBUG>0:
        print ("GDC_FETCH:                                                       \033[1m{:}!!! uberlay mode\033[m".format ( RC ) )     

      walker = os.walk( case_path )
      for root, _, files in walker:
        for f in files:
          if  ( ( fnmatch.fnmatch( f,"*.svs") )  ):                                                  # if we come across an svs file in the case folder     
            already_have_svs_file = True
            if DEBUG>0:
              print ("GDC_FETCH:                                                       \033[1m{:}already have an SVS file for this case ... skipping and moving to next case \033[m".format ( RC ) )

    if already_have_svs_file == False:

      if overlay=="yes":
        if DEBUG>0:
          print ("GDC_FETCH:                                                       \033[1m{:}!!! overlay mode\033[m".format ( RC ) )
          
      if DEBUG>0:
        print( "GDC_FETCH:    \033[1m2a:\033[m requesting file UUIDs for case                 {:}{:}\033[m".format( RC, case )  )


      RESULT, case_files = fetch_case_file_ids   ( RC, DEBUG,                        case,                portal,  file_filter,  uberlay,  overlay, already_have_flag  )
      if RESULT == 1:
        tarfile = download                       ( RC, DEBUG, output_dir, case_path, case,  case_files,   portal                                                       )
        result  = unpack_tarball                 ( RC, DEBUG,             case_path,        tarfile,                                                                   )
        result  = decompress_gz_files            ( RC, DEBUG,             case_path                                                                                    )
        #result  = promote_leaf_files             ( RC, DEBUG,             case_path,                                                                                   )
        result  = setup_and_fill_case_subdirs    ( RC, DEBUG,             case_path                                                                                    )
        result  = delete_unwanted_files          ( RC, DEBUG, output_dir                                                                                               )
        result  = _all_downloaded_ok             ( RC, DEBUG,             case_path                                                                                    ) 
        global_download_counter+=1   
      else:
        pass


    if DEBUG>0:
      print( "\nGDC_FETCH:    all done".format )

#====================================================================================================================================================
# 2a FETCH CASE FILE IDs

def fetch_case_file_ids( RC, DEBUG, case, portal, file_filter, uberlay, overlay, already_have_flag ):

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
  
  if DEBUG>99:
    print( "GDC_FETCH:          response (json list of file ids of hits)  =   {:}{:}\033[m".format(RC, case_files.text ) )

  hits = json.loads( case_files.content.decode("utf-8"))["data"]["hits"]
  
  if DEBUG>0:
    print ( 'GDC_FETCH:          ["hits"]                                  =  \033[32;1m{:}\033[m'.format( hits )   )

  if not len(hits) == 0:                                                                                   # no files match the filter (which is perfectly possible)
    if DEBUG>0:
      print ( 'GDC_FETCH:          already_have_flag:                           \033[32;1m{:}\033[m'.format( already_have_flag )   )

    if ( overlay=="yes" ) | ( uberlay=="yes" ) :
      if Path( already_have_flag ).is_dir():	                                                           # uberlay or overlay mode and there are new files, so delete the already have flag to ensure integrity checking for this download	      
        os.rmdir ( already_have_flag )

      if DEBUG>0:
        print ( 'GDC_FETCH:          new files to be downloaded so deleting:      \033[32;1m{:}\033[m'.format( already_have_flag )   )

    return SUCCESS, case_files

  else:
    if DEBUG>0:
      print ( 'GDC_FETCH:          no new files for this case - skipping' ) 
       
    return FAIL, case_files


#====================================================================================================================================================
# 2b DOWNLOAD CASE FILES

def download( RC, DEBUG, output_dir, case_path, case, case_files, portal ):
    
  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2b:\033[m about to populate file UUID download list and request files" )
    
  file_uuid_list = []

  # (i) Populate the download list with the file_ids from the previous query
  for file_entry in json.loads(case_files.content.decode("utf-8"))["data"]["hits"]:

    file_uuid_list.append(file_entry["file_id"])
      
  if DEBUG>0:
    print( "GDC_FETCH:          files to be downloaded for this case     =  {:}{:}\033[m".format( RC, file_uuid_list) )


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
  
  download_file_name = re.findall("filename=(.+)", response_head_cd)[0]                                    # extract filename from HTTP response header
 
  if DEBUG>9:
    print( "GDC_FETCH:          response.headers[Content-Disposition]     = {:}'{:}'\033[m".format( RC, download_file_name ) )
    print( "GDC_FETCH:          download_file_subdir_name                 = {:}'{:}'\033[m".format( RC, case_path ) )

  if not os.path.isdir ( case_path ):
    os.makedirs( case_path )
  
  download_file_fq_name = "{:}/{:}".format( case_path, download_file_name )

  if DEBUG>9:
    print( "GDC_FETCH:          download_file_fq_name                     = {:}'{:}'\033[m".format( RC, download_file_fq_name ) )

  with open(download_file_fq_name, "wb") as output_file_handle:                                            # save the downloaded file
    output_file_handle.write(response.content)


  # if it's not already a tarball we will turn it into one to allow for uniform processing
  if not download_file_fq_name.endswith("tar.gz"):                                                         
    
    if DEBUG>0:
      print( "GDC_FETCH:            SINGLETON: will create tarball from:      {:}'{:}'\033[m".format( RC, download_file_fq_name ) )

    standard_name   = "STANDARD_NAME.tar" 
    tarfile_fq_name = "{:}{:}".format( case_path,  standard_name      ) 
    arcpath         = "{:}/{:}/{:}".format( output_dir, case, download_file_name )

    if DEBUG>0:
      print( "GDC_FETCH:            SINGLETON: arcpath:                       {:}'{:}'\033[m".format( RC, arcpath ) )

    try:
      tf = tarfile.open( tarfile_fq_name, 'x:gz' )                                                         # create new tarfile called STANDARD_NAME.tar in case_path
    except Exception:
      pass

    try:
      tf.add( download_file_fq_name, arcname=arcpath )
    except Exception:
      pass
 
    if DEBUG>0:
      print( "GDC_FETCH:            SINGLETON: tarball created ok:            {:}'{:}'\033[m".format( RC, standard_name ) )
        
    try:
      tf.close()
    except Exception:
      pass

    try:
      os.remove( download_file_fq_name )
      if DEBUG>0:
        print( "GDC_FETCH:            SINGLETON: now deleting:                  {:}'{:}'\033[m".format( RC, tarfile_fq_name ) )
    except Exception:
      pass

    return standard_name                                                                                   # the case where there was no tarball, so we created one
  
  return download_file_name                                                                                # the case where a tarball was downloaded


#====================================================================================================================================================
# 2c UNPACK TARBALL
 
def unpack_tarball ( RC, DEBUG, case_path, tarfile_name ):

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

  if DEBUG>0:
    print( "GDC_FETCH:          tarball unpacked ok\033[m" )

  #try:
  #  os.remove( tarfile_fq_name )
   # if DEBUG>0:
    #  print( "GDC_FETCH:    \033[1m2c:\033[m now deleting tarball" )
  #except Exception:
   # pass

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
     
        if DEBUG>0:                                                                                        # delete the gz file
          print ( "GDC_FETCH:          about to delete                             {:}'{:}'\033[m".format( RC, fq_name )      )
        try:
          os.remove( fq_name )
        except Exception:
          pass
  return SUCCESS
  
#====================================================================================================================================================
# 2e PROMOTE LEAF FILES UP INTO CASE DIRECTORY

# try using bash: 
#   
# first issue IFS=$'\n' 
#
# then either:
#
#   for x in * ; do mv "$x"/*/* "$x"/ ; done   << didn't try this version
#
# or
#
#    for x in */* ; do [ -d $x ] && ( cd $x ; pwd ; mv * .. ; cd ../.. ) ; done    << this is the version I used and it worked. Had to apply it two times
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
def promote_leaf_files( RC, DEBUG, case_path  ):

  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2e:\033[m about to promote leaf files of interest up into the case directory" )

  walker = os.walk( case_path, topdown=True )
  for root, _, files in walker:

    for f in files:

      fq_name = "{:}{:}".format( case_path, f )

      if DEBUG>0:
        print ( "GDC_FETCH:            case_path                               = {:}'{:}'\033[m".format( RC, case_path )      )
        print ( "GDC_FETCH:            f                                       = {:}'{:}'\033[m".format( RC, f )      )
                                                                          
      if  ( ( fnmatch.fnmatch( f, "MANIFEST.*" ) )  ):                                                    # have to delete, because "MANIFEST.TXT" always only one level down, not two levels down; and there are text files that we DON'T want to delete, so we can't pick up later in the general clean up function
        if DEBUG>0:
          print ( "GDC_FETCH:          about to delete                           = {:}'{:}'\033[m".format( RC, fq_name )      )
        try:
          os.remove( fq_name )
        except Exception:
          pass
        
      elif ( fq_name.endswith("*.tar") ) | ( fq_name.endswith("*.gz") ):                                   # these will be deleted later anyway, and we don't want them promoted up above output_dir
        pass

      else: 
        fq_name      = "{:}/{:}".format( root, f )
        if DEBUG>0:
          print ( "GDC_FETCH:          moving file up a level: filename          = {:}'{:}'\033[m".format( RC, fq_name )      )
        move_to_name = "{:}/../{:}".format( root, f )
        if DEBUG>0:
          print ( "GDC_FETCH:          moving file up a level: new name          = {:}'{:}\033[m".format( RC, move_to_name ) )     
  
        sh.move( fq_name, move_to_name )

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

        try:
          new_dir = os.mkdir( new_dir_name )                                                               # if necessary, create a new case-level subdirectory with unique numeric suffix for this SVS file
        except OSError:
          pass

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
# 2g DELETE UNWANTED FILES AND EMPTY DIRECTORIES

def delete_unwanted_files( RC, DEBUG, output_dir ):
	
  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2g:\033[m about to delete temp files and directories" )
    

  if DEBUG>0:
        print( "GDC_FETCH:          root directory for deleting is:              \033[1m{:}{:}\033[m".format( RC, output_dir ) )

  walker = os.walk( output_dir, topdown=False )

  for root, dirs, files in walker:

    for f in files:
      fqf = root + '/' + f
      if DEBUG>99:
        print( "GDC_FETCH:          examining file:                              {:}{:}\033[m".format( RC, fqf ) )
      if ( f.endswith("tar") ) | ( f.endswith("gz") ): 
        try:
          if DEBUG>99:
            print( "GDC_FETCH:              will delete                              {:}{:}\033[m".format( RC, fqf ) )
          os.remove( fqf )
        except:
          pass

    for d in dirs:
      fqd = root + '/' + d
      if DEBUG>99:
        print( "GDC_FETCH:          examining directory:                         {:}{:}\033[m".format( RC, fqd ) )
      if( os.path.isdir( fqd )):
        if not d.endswith( already_have_suffix ): 
          try:
            if DEBUG>99:
              print( "GDC_FETCH:          will delete if empty:                        {:}{:}\033[m".format( RC, fqd ) )
            os.rmdir( fqd )
          except:
            pass

    
  return SUCCESS

#====================================================================================================================================================
#  2h  create a new case level subdirectory named to indicate the case was handled successfully

def _all_downloaded_ok( RC, DEBUG, case_path ):

  if DEBUG>0:
    print( "GDC_FETCH:    \033[1m2h:\033[m about to create new case level subdirectory, named to indicate the case was handled successfully" )
    
    os.mkdir( case_path[:-1] + already_have_suffix )                   


  return SUCCESS
  
#====================================================================================================================================================
      
if __name__ == '__main__':
	
    p = argparse.ArgumentParser()

    p.add_argument('--debug',                type=int, default=1)
    p.add_argument('--gdc_portal',           type=str, default="main")
    p.add_argument('--case_filter',          type=str, default="dlbc_case_filter")
    p.add_argument('--file_filter',          type=str, default="dlbc_file_filter_just_rna-seq")
    p.add_argument('--max_cases',            type=int, default=5)
    p.add_argument('--max_files',            type=int, default=10)
    p.add_argument('--global_max_downloads', type=int, default=200)
    p.add_argument('--output_dir',           type=str, default="out")
    p.add_argument('--uberlay',              type=str, default="no")
    p.add_argument('--overlay',              type=str, default="no")
    p.add_argument('--delete_compressed',    type=str, default="yes")
    p.add_argument('--cleanup',              type=str, default="no")

    args, _ = p.parse_known_args()

    main(args)

