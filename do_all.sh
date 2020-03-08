#!/bin/bash

source conf/variables.sh

echo "===> STARTING"
echo "=====> STEP 1 OF 7: DELETING EXTRANEOUS FILES FROM DATA TREE"

echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
echo "DO_ALL.SH: INFO: recursively deleting residual gz files:  '${FLAG_DIR_SUFFIX}'"
find ${DATA_DIR} -type f -name "*.gz"                      -exec rm    {} \; 
echo "DO_ALL.SH: INFO: recursively deleting residual tar files:  '${FLAG_DIR_SUFFIX}'"
find ${DATA_DIR} -type f -name "*.tar"                     -exec rm    {} \;
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} \;
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -exec rm -f {} \;

tree ${DATA_DIR}


sleep ${SLEEP_TIME}
cd ${BASE_DIR}

echo "=====> STEP 2 OF 7: GENERATING TILES FROM SVS IMAGES"
sleep ${SLEEP_TIME}
cd ${BASE_DIR}
./start.sh

echo "=====> STEP 3 OF 7: EXTRACTING RNA EXPRESSION INFORMATION, SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
python process_rna_exp.py "--data_dir="${DATA_DIR} "--rna_file_suffix="${RNA_FILE_SUFFIX} "--rna_exp_column="${RNA_EXP_COLUMN} "--rna_numpy_filename="${RNA_NUMPY_FILENAME}

echo "=====> STEP 4 OF 7: PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  

echo "=====> STEP 5 OF 7: CONVERTING DATA INTO FORMAT SUITABLE FOR PYTORCH TO USE (DICTIONARY OF TORCH TENSORS)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
python -m ${NN_DATASET_HELPER_APPLICATION_NAME} ${DATA_DIR} ${N_SAMPLES} ${TILE_SIZE} ${TILES_PER_IMAGE} ${RNA_NUMPY_FILENAME} ${CLASS_NUMPY_FILENAME}  

NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}


echo "=====> STEP 6 OF 7: CLEAN DATA TREE"
sleep ${SLEEP_TIME}
echo "DO_ALL.SH: INFO: deleting any residual empty subdirectories"
find ${DATA_DIR} -type d -empty -delete +
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${MASK_FILE_NAME_SUFFIX}'"
find ${DATA_DIR} -type f -name ${MASK_FILE_NAME_SUFFIX}    -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
find ${DATA_DIR} -type f -name ${RESIZED_FILE_NAME_SUFFIX} -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${RNA_FILE_SUFFIX}'"
find ${DATA_DIR} -type f -name ${RNA_FILE_SUFFIX}          -exec rm -f {} +



echo "=====> STEP 7 OF 7: TRAINING"
sleep ${SLEEP_TIME}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} --dataset=${DATASET} --n_samples=${N_SAMPLES} --nn_mode=${NN_MODE} --nn_type=${NN_TYPE} --input_mode=${INPUT_MODE} --n_epochs=${N_EPOCHS} --n_genes=${N_GENES} --n_tiles=${TILES_PER_IMAGE} --whiteness=${MAXIMUM_PERMITTED_WHITENESS} --greyness=${MINIMUM_PERMITTED_GREYSCALE_RANGE} --colour_norm=${COLOUR_NORMALIZATION} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}


echo "===> FINISHED "
cd ${BASE_DIR}
