#!/bin/bash

source conf/variables.sh

echo "===> STARTING"
echo "=====> STEP 1 OF 7: DELETING EXTRANEOUS FILES FROM DATA TREE"

echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} + 
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -exec rm -f {} +

SLEEP_TIME=2
sleep ${SLEEP_TIME}
cd ${BASE_DIR}

echo "=====> STEP 2 OF 7: GENERATING TILES FROM SVS IMAGES"
sleep ${SLEEP_TIME}
cd ${BASE_DIR}
./start.sh

echo "=====> STEP 3 OF 7: EXTRACTING RNA EXPRESSION INFORMATION, SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
python process_rna_exp.py ${DATA_DIR} ${RNA_FILE_SUFFIX} ${RNA_EXP_COLUMN} ${RNA_NUMPY_FILENAME}

echo "=====> STEP 4 OF 7: PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  

echo "=====> STEP 5 OF 7: CONVERTING DATA INTO FORMAT SUITABLE FOR PYTORCH TO USE (DICTIONARY OF TORCH TENSORS)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
python -m ${NN_DATASET_HELPER_APPLICATION_NAME}  ${DATA_DIR} ${TILE_SIZE} ${TILES_TO_GENERATE_PER_SVS} ${RNA_NUMPY_FILENAME} ${CLASS_NUMPY_FILENAME}  

NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}


echo "=====> STEP 6 OF 7: DELETING INTERMEDIATE FILES FROM DATA TREE"
sleep ${SLEEP_TIME}
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${MASK_FILE_NAME_SUFFIX}'"
find ${DATA_DIR} -type f -name ${MASK_FILE_NAME_SUFFIX}    -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
find ${DATA_DIR} -type f -name ${RESIZED_FILE_NAME_SUFFIX} -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${RNA_FILE_SUFFIX}'"
find ${DATA_DIR} -type f -name ${RNA_FILE_SUFFIX}          -exec rm -f {} +

echo "=====> STEP 7 OF 7: TRAINING"
sleep ${SLEEP_TIME}
python ${NN_MAIN_APPLICATION_NAME} --mode=${MODE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}


echo "===> FINISHED "
cd ${BASE_DIR}
