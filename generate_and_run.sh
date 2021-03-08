#!/bin/bash

# exit if any command fails
# set -e

echo ""
echo ""
echo ""


export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

MULTIMODE="NONE"                                                         # possibly changed by user '-m' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                              # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                     # possibly changed by user '-v' argument if required, but it needs an initial value

MULTIMODE="NONE"                                                         # possibly changed by user '-m' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                              # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                     # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"

while getopts c:d:i:m:p:t:r:v: option
  do
    case "${option}"
    in
    c) CASES=${OPTARG};;                                                     # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    d) DATASET=${OPTARG};;                                                   # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    i) INPUT_MODE=${OPTARG};;                                                # supported: image, rna, image_rna
    m) MULTIMODE=${OPTARG};;                                                 # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    p) PRETRAIN=${OPTARG};;                                                  # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    t) JUST_TEST=${OPTARG};;                                                 # 'test'  or nothing
    r) REGEN=${OPTARG};;                                                     # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    v) DIVIDE_CASES=${OPTARG};;                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    esac
  done

#~ echo ${CASES}
#~ echo ${DATASET}
#~ echo ${INPUT_MODE}
#~ echo ${MULTIMODE}
#~ echo ${JUST_TEST}
#~ echo ${REGEN}

source conf/variables.sh ${DATASET}

echo "=====> STEP 1 OF 1: RUNNING THE NETWORK (DATASET DIRECTORY WILL NOT BE CLEANED; TILING WILL NOT BE PERFORMED; PYTORCH DATASET WILL NOT BE REGENERATED)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --multimode ${MULTIMODE} --use_tiler ${USE_TILER} --just_profile ${JUST_PROFILE} --just_test ${JUST_TEST} --skip_tiling 'True' --skip_generation 'False' \
--dataset ${DATASET} --cases ${CASES} --data_dir ${DATA_DIR} --data_source ${DATA_SOURCE} --divide_cases ${DIVIDE_CASES} --cases_reserved_for_image_rna ${CASES_RESERVED_FOR_IMAGE_RNA} \
--global_data ${GLOBAL_DATA} --mapping_file_name ${MAPPING_FILE_NAME} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--ddp ${DDP} --use_autoencoder_output ${USE_AUTOENCODER_OUTPUT} --pretrain ${PRETRAIN} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  --use_unfiltered_data ${USE_UNFILTERED_DATA} --remove_low_expression_genes  ${REMOVE_LOW_EXPRESSION_GENES} \
--embedding_file_suffix_rna ${EMBEDDING_FILE_SUFFIX_RNA} --embedding_file_suffix_image ${EMBEDDING_FILE_SUFFIX_IMAGE} --embedding_file_suffix_image_rna ${EMBEDDING_FILE_SUFFIX_IMAGE_RNA} \
--low_expression_threshold ${LOW_EXPRESSION_THRESHOLD} --remove_unexpressed_genes ${REMOVE_UNEXPRESSED_GENES} --target_genes_reference_file ${TARGET_GENES_REFERENCE_FILE} \
--a_d_use_cupy ${A_D_USE_CUPY} --cov_threshold ${COV_THRESHOLD} --cov_uq_threshold ${COV_UQ_THRESHOLD} --cutoff_percentile ${CUTOFF_PERCENTILE} \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --highest_class_number ${HIGHEST_CLASS_NUMBER} \
--nn_mode ${NN_MODE} --use_same_seed ${USE_SAME_SEED} --nn_type_img ${NN_TYPE_IMG} --nn_type_rna ${NN_TYPE_RNA}  \
--nn_dense_dropout_1 ${NN_DENSE_DROPOUT_1} --nn_dense_dropout_2 ${NN_DENSE_DROPOUT_2} \
--encoder_activation ${ENCODER_ACTIVATION} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} --n_tests ${N_TESTS} --final_test_batch_size ${FINAL_TEST_BATCH_SIZE} \
--gene_data_norm ${GENE_DATA_NORM} --gene_data_transform ${GENE_DATA_TRANSFORM} --gene_embed_dim ${GENE_EMBED_DIM} --hidden_layer_neurons ${HIDDEN_LAYER_NEURONS} --hidden_layer_encoder_topology ${HIDDEN_LAYER_ENCODER_TOPOLOGY} \
--cancer_type ${CANCER_TYPE} --cancer_type_long ${CANCER_TYPE_LONG} --class_names ${CLASS_NAMES} --long_class_names ${LONG_CLASS_NAMES} --class_colours ${CLASS_COLOURS} --colour_map ${COLOUR_MAP} \
--n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --zoom_out_mags ${ZOOM_OUT_MAGS} --zoom_out_prob ${ZOOM_OUT_PROB} \
--n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} \
--greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} --label_swap_perunit ${LABEL_SWAP_PERUNIT} \
--target_tile_offset ${TARGET_TILE_OFFSET} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}  --points_to_sample ${POINTS_TO_SAMPLE} \
--show_rows ${SHOW_ROWS} --show_cols ${SHOW_COLS} --figure_width ${FIGURE_WIDTH} --figure_height ${FIGURE_HEIGHT} --annotated_tiles ${ANNOTATED_TILES} --supergrid_size ${SUPERGRID_SIZE} \
--patch_points_to_sample ${PATCH_POINTS_TO_SAMPLE} --scattergram ${SCATTERGRAM} --box_plot ${BOX_PLOT} --minimum_job_size ${MINIMUM_JOB_SIZE} --show_patch_images ${SHOW_PATCH_IMAGES} \
--bar_chart_x_labels=${BAR_CHART_X_LABELS} --bar_chart_sort_hi_lo=${BAR_CHART_SORT_HI_LO}  --bar_chart_show_all=${BAR_CHART_SHOW_ALL} \
--probs_matrix ${PROBS_MATRIX} --probs_matrix_interpolation ${PROBS_MATRIX_INTERPOLATION} 

