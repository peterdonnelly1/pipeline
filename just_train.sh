#!/bin/bash


source conf/variables.sh

cd ${NN_APPLICATION_PATH}

echo "=====> TRAINING"
sleep 1
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} --dataset=${DATASET} --n_samples=${N_SAMPLES} --nn_mode=${NN_MODE} --nn_type=${NN_TYPE} --input_mode=${INPUT_MODE} --n_epochs=${N_EPOCHS} --n_genes=${N_GENES} --n_tiles=${TILES_PER_IMAGE} --whiteness=${MAXIMUM_PERMITTED_WHITENESS} --greyness=${MINIMUM_PERMITTED_GREYSCALE_RANGE} --batch_size=${BATCH_SIZE} --latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES}

cd ${BASE_DIR}
