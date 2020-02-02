#!/bin/bash
clear
source conf/variables.sh

echo "===> STARTING"
echo "=====> STEP 1 OF 8: DELETING UNECESSARY FILES FROM DATA TREE"
find . -type d -name '*_all_downloaded_ok' -exec rmdir {} + 

SLEEP_TIME=4

sleep ${SLEEP_TIME}

cd ${BASE_DIR}

echo "=====> STEP 2 OF 8: GENERATING TILES FROM SVS IMAGES"
sleep ${SLEEP_TIME}

cd ${BASE_DIR}
./start.sh

echo "=====> STEP 3 OF 8: PLACING COPY OF ENTIRE TRUTH VECTOR (e.g. 'TISSUE CLASS') INTO EACH CASE SUB-DIRECTORY AS FILE 'class.csv' FOR SUBSEQUENT USE"
sleep ${SLEEP_TIME}

./setup_class_files.sh

echo "=====> STEP 4 OF 8: EXTRACTING RNA EXPRESSION & CLASS INFORMATION, SAVING AS NUMPY FILE"
sleep ${SLEEP_TIME}

python process_rna_seq.py ${DATA_DIR} ${RNA_FILE_PATTERN} ${RNA_NUMPY_FILENAME}

echo "=====> STEP 5 OF 8: PRE-PROCESSING CLASS (GROUND TRUTH) DATA AND SAVING AS NUMPY FILE"
sleep ${SLEEP_TIME}

python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  

echo "=====> STEP 5 OF 8: CONVERTING DATA INTO FORMAT SUITABLE FOR PYTORCH TO USE (DICTIONARY OF TORCH TENSORS)"
sleep ${SLEEP_TIME}

cd ${NN_APPLICATION_PATH}
python -m ${NN_DATASET_HELPER_APPLICATION_NAME}  ${DATA_DIR} ${TILE_SIZE} ${TILES_TO_GENERATE_PER_SVS} ${RNA_NUMPY_FILENAME} ${TISSUE_NUMPY_FILENAME}  


NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}

echo "=====> STEP 6 OF 8: DELETING EXTRANEOUS FILES FROM DATA TREE"
find ${DATA_DIR} -type f -name '*_mask.png' -exec rm {} +
find ${DATA_DIR} -type f -name '*_resized.png' -exec rm {} + 
find ${DATA_DIR} -type f -name '*.FPKM-UQ.txt' -exec rm {} + 

echo "=====> STEP 7 OF 8: TRAINING"
sleep ${SLEEP_TIME}

python ${NN_MAIN_APPLICATION_NAME} --mode=${MODE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}


echo "===> FINISHED "
cd ${BASE_DIR}
