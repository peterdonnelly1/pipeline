#!/bin/bash

source conf/variables.sh

echo "===> STARTING"

echo "=====> STEP 1 OF 2: CONVERTING DATA INTO FORMAT SUITABLE FOR PYTORCH TO USE (DICTIONARY OF TORCH TENSORS)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
python -m ${NN_DATASET_HELPER_APPLICATION_NAME} ${DATA_DIR} ${N_SAMPLES} ${TILE_SIZE} ${TILES_PER_IMAGE} ${RNA_NUMPY_FILENAME} ${CLASS_NUMPY_FILENAME}  


echo "=====> STEP 2 OF 2: TRAINING"
sleep ${SLEEP_TIME}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} --dataset=${DATASET} --n_samples=${N_SAMPLES} --nn_mode=${NN_MODE} --nn_type=${NN_TYPE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}


echo "===> FINISHED "
cd ${BASE_DIR}
