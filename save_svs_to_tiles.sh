#!/bin/bash

source conf/variables.sh

bold=$(tput bold)
normal=$(tput sgr0)


stdbuf -i0 -o0 -e0 echo

MY_THREAD_NUMBER=$1                      # my thread number
NUMBER_OF_THREADS=$2                     # the total number of threads (= shell scripts, which will run in parallel)

DIR_N=0

#echo "SVS_PATH ="${DATA_DIR}

#for dir in ${SVS_PATH}/*/; do    # list directories
#    echo "absolute path (dir)      =    "${dir}
#    dir=${dir%*/}      # remove the trailing "/"
#    echo "just file (directory) name (dir##*/) =   "${dir##*/}    # print everything after the final "/"
#done


cd ${DATA_DIR}
    
# skip over files that other threads will handle

#for dir in /home/peter/git/pipeline/output/*/; do
for dir in ${DATA_DIR}/*/; do
  
#  printf "  SAVE_SVS_TO_TILES.SH: INFO: now descending into      "$dir"\n"

  if [ -d $dir ]; then
    
    DIR_N=$((DIR_N+1))
    
    if [ $((DIR_N % NUMBER_OF_THREADS)) -ne ${MY_THREAD_NUMBER} ]; then continue; fi

    cd ${dir}
    
#    printf "  SAVE_SVS_TO_TILES.SH: INFO: pwd                    = "
#    pwd
#    printf "\n"
    
    SLIDES_PROCESSED=$( echo $((MY_THREAD_NUMBER+1 * DIR_N)) )
    
#    printf  "  SAVE_SVS_TO_TILES.SH: INFO:  now processing processing slide number --------------------------------------------------------------------------> " 
#    printf  ${MY_THREAD_NUMBER}
#    printf  "   " 
#    printf  ${DIR_N}
#    printf  "   " 
#    printf  ${SLIDES_PROCESSED}
#    printf  "\n" 
    
    for SVS in ${dir}*.svs; do

    SLIDE_FILE_NAME=`echo ${SVS} | awk -F'/' '{print $NF}'`                                                          # extract just the file name from the fully qualified name and save
        
#    printf "  SAVE_SVS_TO_TILES.SH: INFO: slide file name        = "${SVS}"\n"
    
    # (1) make and save mask for this SVS file
#        printf   "  SAVE_SVS_TO_TILES.SH: INFO: about to extract background for "${bold}${SVS}${normal} "\n"
        python -u ${BASE_DIR}/background_mask.py ${SLIDE_FILE_NAME} ${dir}                                           # start Python in unbuffered mode
 #       printf "  SAVE_SVS_TO_TILES.SH: INFO: done  extracting  background for "${bold}${SVS}${normal} "\n"
    
      done
    
    # (2) make and save tiles (possibly hundreds or thousands) for this SVS file
    python ${BASE_DIR}/save_svs_to_tiles.py "--slide_name="${SLIDE_FILE_NAME} "--greyness="${MINIMUM_PERMITTED_GREYSCALE_RANGE} "--file_dir="${dir} "--my_thread="${MY_THREAD_NUMBER} "--n_tiles="${TILES_PER_IMAGE}  \
"--rand_tiles="${RANDOM_TILES} "--tile_size="${TILE_SIZE} "--min_uniques="${MINIMUM_PERMITTED_UNIQUE_VALUES}  \
"--stain_norm="${STAIN_NORMALIZATION} "--stain_norm_target="${STAIN_NORM_TARGET} "--min_tile_sd="${MIN_TILE_SD}  "--points_to_sample="${POINTS_TO_SAMPLE}

#    printf "  SAVE_SVS_TO_TILES.SH: INFO: done tiling image:    " ${SLIDE_FILE_NAME} "\n"
    
    if [ $? -ne 0 ]; then
        echo "  SAVE_SVS_TO_TILES.SH: INFO: failed extracting tiles for " ${SVS}
        rm -rf ${DATA_DIR}/${SVS}
    #else
        #cd ./stain_norm_python
        #python color_normalize_single_folder.py ${DATA_DIR}/${SVS}
        #cd ../
        #wait
        #touch ${dir}/extraction_done.txt
    
    
    fi

    cd ..

  fi
 
 done
  
exit 0;