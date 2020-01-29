#!/bin/bash
clear
source conf/variables.sh

echo "===> STARTING"
echo "=====> STEP 1 OF 4: GENERATING TILES FROM SVS IMAGES"

sleep 1

cd ${BASE_DIR}
./start.sh

echo "=====> STEP 2 OF 4: EXTRACTING RELEVANT COLUMNS FROM RNA EXPRESSION FILES, CONVERTING TO NUMPY AND SAVING IN NUMPY FRIENDLY FILE FORMAT"
sleep 1
python extract_rna_column_convert_to_python_friendly.py ${EXPERIMENT_DIR} ${RNA_FILE_PATTERN} ${RNA_NUMPY_FILENAME} ${TISSUE_NUMPY_FILENAME}

#find ${PATCH_PATH} -name *.results
#find ${PATCH_PATH} -name *.results | wc -l
#find ${PATCH_PATH} -name *.npy
#find ${PATCH_PATH} -name *.npy     | wc -l

echo "=====> STEP 3 OF 4: CONVERTING DATA INTO FORMAT SUITABLE FOR PYTORCH TO USE (DICTIONARY OF TORCH TENSORS)"
sleep 1
cd ${NN_APPLICATION_PATH}
python -m ${NN_DATASET_HELPER_APPLICATION_NAME}  ${EXPERIMENT_DIR} ${TILE_SIZE} ${TILES_TO_GENERATE_PER_SVS} ${RNA_NUMPY_FILENAME} ${TISSUE_NUMPY_FILENAME}  

echo "=====> STEP 4 OF 4: TRAINING"
sleep 1
python ${NN_MAIN_APPLICATION_NAME} --dataset=${EXPERIMENT} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}


echo "===> FINISHED "
cd ${BASE_DIR}

NUMBER_OF_TILES=$(find ${PATCH_PATH} -name *${TILE_SIZE}.png | wc -l)
echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}
