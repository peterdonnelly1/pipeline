#!/bin/bash

# exit if any command fails
# set -e

DATASET="$1"
source conf/variables.sh ${DATASET}

export MKL_DEBUG_CPU_TYPE=5

echo "===> STARTING"
if [ "$2" == "regen" ]; 
  then
    echo "=====> STEP 0 OF 4: REGENERATING DATASET FOLDER (THIS CAN TAKE UP TO SEVERAL MINUTES)"
    rm -rf ${DATA_DIR}
    rsync -ah --info=progress2 $1/ ${DATA_DIR}
  else
    echo "=====> STEP 0 OF 4: DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
    echo "DO_ALL.SH: INFO: deleting all empty subdirectories under '${DATA_DIR}'"
    find ${DATA_DIR} -type d -empty -delete +
    echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
    echo "DO_ALL.SH: INFO: recursively deleting residual gz  files:  '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type f -name "*.gz"                      -exec rm    {} \; 
    echo "DO_ALL.SH: INFO: recursively deleting residual tar files:  '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type f -name "*.tar"                     -exec rm    {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
    find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
    find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files (tiles)  matching this pattern:  '*.png'       <<< this can take some time"
    find ${DATA_DIR} -type f -name *.png                       -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${MASK_FILE_NAME_SUFFIX}'"
    find ${DATA_DIR} -type f -name ${MASK_FILE_NAME_SUFFIX}    -exec rm -f {} +
    echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
    find ${DATA_DIR} -type f -name ${RESIZED_FILE_NAME_SUFFIX} -exec rm -f {} +
fi

tree ${DATA_DIR}

cd ${BASE_DIR}

echo "=====> STEP 1 OF 4: GENERATING TILES FROM SLIDE IMAGES"
if [ ${USE_TILER} == "external" ]; 
  then
    sleep ${SLEEP_TIME}
    ./start.sh
  else
    echo "DO_ALL.SH: INFO:  skipping tile generation in accordance with user parameter 'USE_TILER'"
fi

echo "=====> STEP 2 OF 4: EXTRACTING RNA EXPRESSION INFORMATION AND SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
python process_rna_exp.py "--data_dir="${DATA_DIR} "--rna_file_suffix="${RNA_FILE_SUFFIX} "--rna_exp_column="${RNA_EXP_COLUMN} "--rna_numpy_filename="${RNA_NUMPY_FILENAME}

echo "=====> STEP 3 OF 4: PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
cp $1_global/mapping_file ${DATA_DIR};
python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  

NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}

echo "=====> STEP 4 OF 4: RUNNING THE NETWORK"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode=${INPUT_MODE} --use_tiler=${USE_TILER} --dataset=${DATASET}     --data_dir=${DATA_DIR} --rna_file_name=${RNA_NUMPY_FILENAME} --class_numpy_file_name=${CLASS_NUMPY_FILENAME} \
--nn_mode=${NN_MODE}       --nn_type=${NN_TYPE} --n_samples=${N_SAMPLES} --n_genes=${N_GENES} --class_names ${CLASS_NAMES} --n_tiles=${TILES_PER_IMAGE} --rand_tiles=${RANDOM_TILES} --tile_size=${TILE_SIZE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} \
--latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES} \
--min_uniques=${MINIMUM_PERMITTED_UNIQUE_VALUES} --greyness=${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit=${MAKE_GREY_PERUNIT} \
--target_tile_coords ${TARGET_TILE_COORDS} --stain_norm=${STAIN_NORMALIZATION} --stain_norm_target=${STAIN_NORM_TARGET} \
--min_tile_sd=${MIN_TILE_SD}


cd ${BASE_DIR}

echo "===> FINISHED "
