#!/bin/bash


source conf/variables.sh

cd ${NN_APPLICATION_PATH}

echo "=====> TRAINING"
sleep 1
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} --mode=${MODE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}

echo "===> ALL PRE-PROCESSING FINISHED "
cd ${BASE_DIR}
