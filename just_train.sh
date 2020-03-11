#!/bin/bash


source conf/variables.sh

cd ${NN_APPLICATION_PATH}

echo "=====> TRAINING"
sleep 1
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode=${INPUT_MODE}   --dataset=${DATASET}     --data_dir=${DATA_DIR} --rna_file_name=${RNA_NUMPY_FILENAME} --class_numpy_file_name=${CLASS_NUMPY_FILENAME} \
--nn_mode=${NN_MODE}       --nn_type=${NN_TYPE} --n_samples=${N_SAMPLES} --n_genes=${N_GENES} --n_tiles=${TILES_PER_IMAGE} --tile_size=${TILE_SIZE} --n_epochs=${N_EPOCHS} --batch_size=${BATCH_SIZE} \
--latent_dim=${LATENT_DIM} --max_consecutive_losses=${MAX_CONSECUTIVE_LOSSES} \
--whiteness=${MAXIMUM_PERMITTED_WHITENESS} --greyness=${MINIMUM_PERMITTED_GREYSCALE_RANGE} --colour_norm=${COLOUR_NORMALIZATION} 

cd ${BASE_DIR}
