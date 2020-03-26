#!/bin/bash

source conf/variables.sh

export MKL_DEBUG_CPU_TYPE=5

echo "=====> STEP 4 OF 4: RUNNING THE NETWORK"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --use_tiler ${USE_TILER}  --skip_preprocessing 'False' --skip_generation 'False' --dataset ${DATASET}     --data_dir ${DATA_DIR} --rna_file_name ${RNA_NUMPY_FILENAME} --class_numpy_file_name ${CLASS_NUMPY_FILENAME} \
--nn_mode ${NN_MODE}       --nn_type ${NN_TYPE} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --n_genes ${N_GENES} --class_names ${CLASS_NAMES} --n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} \
--min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} --greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} \
--target_tile_coords ${TARGET_TILE_COORDS} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} \
--min_tile_sd ${MIN_TILE_SD}

cd ${BASE_DIR}
