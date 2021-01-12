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

while getopts c:d:i:m:t:r:v: option
  do
    case "${option}"
    in
    c) CASES=${OPTARG};;                                                     # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    d) DATASET=${OPTARG};;                                                   # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    i) INPUT_MODE=${OPTARG};;                                                # supported: image, rna, image_rna
    m) MULTIMODE=${OPTARG};;                                                 # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
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

echo "=====> STEP 1 OF 2: CLEANING DATASET DIRECTORY"
  #~ echo "=====> DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
  #~ echo "DO_ALL.SH: INFO: deleting all empty subdirectories under '${DATA_DIR}'"
  find ${DATA_DIR} -type d -empty -delete
  #~ echo "DO_ALL.SH: INFO: deleting the 'SUFFICIENT_SLIDES_TILED' flag"        
  rm "${DATA_DIR}/SUFFICIENT_SLIDES_TILED" > /dev/null 2>&1
  #~ echo "DO_ALL.SH: INFO: deleting all 'SLIDE_TILED_FLAG' flags"        
  find ${DATA_DIR} -type f -name "SLIDE_TILED_FLAG"          -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
  find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
  #~ echo "DO_ALL.SH: INFO: recursively deleting residual            '.tar' files"
  find ${DATA_DIR} -type f -name "*.tar"                     -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting residual            '.gz'  files"
  find ${DATA_DIR} -type f -name "*.gz"                      -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting                     '.fqln'            files created in earlier runs"
  find ${DATA_DIR} -type l -name "*.fqln"                    -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting                     'entire_patch.npy' files created in earlier runs"
  find ${DATA_DIR} -type l -name "entire_patch.npy"          -delete 
  #~ echo "DO_ALL.SH: INFO: recursively deleting files               matching this pattern:  '*${RNA_FILE_REDUCED_SUFFIX}'"
  #~ find ${DATA_DIR} -type f -name *${RNA_FILE_REDUCED_SUFFIX} -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting files (embeddings)  matching this pattern:  '*_matched.npy'"
  #~ find ${DATA_DIR} -type f -name "*matched.npy"              -delete

  if [[ ${INPUT_MODE} == 'image' ]]; then
      echo "DO_ALL.SH: INFO: image       mode, so recursively deleting existing image     embedding files ('${EMBEDDING_FILE_SUFFIX_IMAGE}')"
      find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}      -delete
  elif [[ ${INPUT_MODE} == 'rna' ]]; then
      echo "DO_ALL.SH: INFO: rna         mode, so recursively deleting existing rna       embedding files ('${EMBEDDING_FILE_SUFFIX_RNA}')"
      find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}        -delete
  elif [[ ${INPUT_MODE} == "image_rna" ]]; then
      echo "DO_ALL.SH: INFO: 'image_rna' mode, so recursively deleting existing image_rna embedding files ('${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}')"
      find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}  -delete
  fi
  

  if [[ ${MULTIMODE} != 'image_rna' ]]; then
      #~ echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
      #~ find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}          -delete
      echo "DO_ALL.SH: INFO: recursively deleting files (tiles)  matching this pattern:  '*.png'                           <<< for image mode, deleting all the .png files (i.e. tiles) can take quite some time as there can be up to millions of tiles"
      find ${DATA_DIR} -type f -name *.png                       -delete
  fi
  


  RANDOM_TILES="False"
  PCT_TEST=1.0


echo "=====> STEP 2 OF 2: RUNNING THE NETWORK (TILING MAY BE PERFORMED; PYTORCH DATASET WILL BE GENERATED)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --multimode ${MULTIMODE} --use_tiler ${USE_TILER} --just_profile 'False' --just_test 'True' --skip_tiling ${SKIP_TILING} --skip_generation 'False' \
--dataset ${DATASET} --cases ${CASES} --data_dir ${DATA_DIR} --data_source ${DATA_SOURCE} --divide_cases ${DIVIDE_CASES} --cases_reserved_for_image_rna ${CASES_RESERVED_FOR_IMAGE_RNA} \
--global_data ${GLOBAL_DATA} --mapping_file_name ${MAPPING_FILE_NAME} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--ddp ${DDP} --use_autoencoder_output ${USE_AUTOENCODER_OUTPUT} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  --use_unfiltered_data ${USE_UNFILTERED_DATA} --remove_low_expression_genes  ${REMOVE_LOW_EXPRESSION_GENES} \
--embedding_file_suffix_rna ${EMBEDDING_FILE_SUFFIX_RNA} --embedding_file_suffix_image ${EMBEDDING_FILE_SUFFIX_IMAGE} --embedding_file_suffix_image_rna ${EMBEDDING_FILE_SUFFIX_IMAGE_RNA} \
--low_expression_threshold ${LOW_EXPRESSION_THRESHOLD} --remove_unexpressed_genes ${REMOVE_UNEXPRESSED_GENES} \
--a_d_use_cupy ${A_D_USE_CUPY} --cov_threshold ${COV_THRESHOLD} --cov_uq_threshold ${COV_UQ_THRESHOLD} --cutoff_percentile ${CUTOFF_PERCENTILE} \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --nn_mode ${NN_MODE} --use_same_seed ${USE_SAME_SEED} --nn_type_img ${NN_TYPE_IMG} --nn_type_rna ${NN_TYPE_RNA}  \
--nn_dense_dropout_1 ${NN_DENSE_DROPOUT_1} --nn_dense_dropout_2 ${NN_DENSE_DROPOUT_2} \
--encoder_activation ${ENCODER_ACTIVATION} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} --final_test_batch_size ${FINAL_TEST_BATCH_SIZE} \
--gene_data_norm ${GENE_DATA_NORM} --gene_data_transform ${GENE_DATA_TRANSFORM} --gene_embed_dim ${GENE_EMBED_DIM} --hidden_layer_neurons ${HIDDEN_LAYER_NEURONS} --hidden_layer_encoder_topology ${HIDDEN_LAYER_ENCODER_TOPOLOGY} \
--cancer_type ${CANCER_TYPE} --cancer_type_long ${CANCER_TYPE_LONG} --class_names ${CLASS_NAMES} --long_class_names ${LONG_CLASS_NAMES} --class_colours ${CLASS_COLOURS} --colour_map ${COLOUR_MAP} \
--n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} \
--greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} --label_swap_perunit ${LABEL_SWAP_PERUNIT} \
--target_tile_offset ${TARGET_TILE_OFFSET} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}  --points_to_sample ${POINTS_TO_SAMPLE} \
--show_rows ${SHOW_ROWS} --show_cols ${SHOW_COLS} --figure_width ${FIGURE_WIDTH} --figure_height ${FIGURE_HEIGHT} --annotated_tiles ${ANNOTATED_TILES} --supergrid_size ${SUPERGRID_SIZE} \
--patch_points_to_sample ${PATCH_POINTS_TO_SAMPLE} --scattergram ${SCATTERGRAM} --box_plot ${BOX_PLOT} --minimum_job_size ${MINIMUM_JOB_SIZE} --show_patch_images ${SHOW_PATCH_IMAGES} \
--bar_chart_x_labels=${BAR_CHART_X_LABELS} --bar_chart_sort_hi_lo=${BAR_CHART_SORT_HI_LO} \
--probs_matrix ${PROBS_MATRIX} --probs_matrix_interpolation ${PROBS_MATRIX_INTERPOLATION} 
cd ${BASE_DIR}
