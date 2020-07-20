#!/bin/bash

source conf/variables.sh

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

echo "=====> DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
echo "DO_ALL.SH: INFO: deleting all empty subdirectories under                     '${DATA_DIR}'"
find ${DATA_DIR} -type d -empty -delete
echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
echo "DO_ALL.SH: INFO: recursively deleting residual .tar files"
find ${DATA_DIR} -type f -name "*.tar"                     -exec rm    {} \;
echo "DO_ALL.SH: INFO: recursively deleting residual .gz  files"
find ${DATA_DIR} -type f -name "*.gz"                      -exec rm    {} \;
#echo "DO_ALL.SH: INFO: recursively deleting '.fqln'            files created in earlier runs"
#find ${DATA_DIR} -type l -name "*.fqln"                    -exec rm    {} \;
echo "DO_ALL.SH: INFO: recursively deleting 'entire_patch.npy' files created in earlier runs"
find ${DATA_DIR} -type l -name "entire_patch.npy"          -exec rm    {} \; 
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} \;
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '*${RNA_FILE_REDUCED_SUFFIX}'"
find ${DATA_DIR} -type f -name *${RNA_FILE_REDUCED_SUFFIX}          -exec rm -f {} \;
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
find ${DATA_DIR} -type f -name ${MASK_FILE_NAME_SUFFIX}    -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
find ${DATA_DIR} -type f -name ${RESIZED_FILE_NAME_SUFFIX} -exec rm -f {} +
echo "DO_ALL.SH: INFO: recursively deleting files (tiles)  matching this pattern:  '*.png'                            <<< for image mode, deleting all the .png files (i.e. tiles) can take quite some time as their can be up to hundreds of thousands"
find ${DATA_DIR} -type f -name *.png                       -exec rm -f {} \;
RANDOM_TILES="False"
PCT_TEST=1.0


echo "=====> STEP 4 OF 4: LAUNCHING THE APPLICATION"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --use_tiler ${USE_TILER} --just_profile 'False' --just_test 'True' --skip_preprocessing 'False' --skip_generation 'False' --rand_tiles ${RANDOM_TILES} --dataset ${DATASET} --data_dir ${DATA_DIR} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  --use_unfiltered_data ${USE_UNFILTERED_DATA} \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --nn_mode ${NN_MODE} --use_same_seed ${USE_SAME_SEED} --nn_type ${NN_TYPE} --nn_dense_dropout_1 ${NN_DENSE_DROPOUT_1} --nn_dense_dropout_2 ${NN_DENSE_DROPOUT_2} \
--optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} \
--n_genes ${N_GENES} --gene_data_norm ${GENE_DATA_NORM} --gene_data_transform ${GENE_DATA_TRANSFORM} \
--cancer_type ${CANCER_TYPE} --cancer_type_long ${CANCER_TYPE_LONG} --class_names ${CLASS_NAMES} --long_class_names ${LONG_CLASS_NAMES} --class_colours ${CLASS_COLOURS} \
--n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} --greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} \
--target_tile_offset ${TARGET_TILE_OFFSET} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}  --points_to_sample ${POINTS_TO_SAMPLE} \
--figure_width ${FIGURE_WIDTH} --figure_height ${FIGURE_HEIGHT} --annotated_tiles ${ANNOTATED_TILES} --supergrid_size ${SUPERGRID_SIZE} --patch_points_to_sample ${PATCH_POINTS_TO_SAMPLE} --scattergram ${SCATTERGRAM} --show_patch_images ${SHOW_PATCH_IMAGES} \
--probs_matrix ${PROBS_MATRIX} --probs_matrix_interpolation ${PROBS_MATRIX_INTERPOLATION} 
cd ${BASE_DIR}
