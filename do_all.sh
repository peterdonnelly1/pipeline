#!/bin/bash

# exit if any command fails
# set -e

source conf/variables.sh ${DATASET}

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

echo "===> STARTING"

if [[ ${SKIP_TILING} == "False" ]]; 
  then
    if [[ "$3" == "regen" ]]; 
      then
        echo "=====> STEP 1 OF 6: REGENERATING DATASET FOLDER (THIS CAN TAKE UP TO SEVERAL MINUTES)"
        rm -rf ${DATA_DIR}
        rsync -ah --info=progress2 $1/ ${DATA_DIR}
      else
        echo "=====> STEP 1 OF 6: DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
        echo "DO_ALL.SH: INFO: deleting all empty sub-directories under                     '${DATA_DIR}'"
        find ${DATA_DIR} -type d -empty -delete
        echo "DO_ALL.SH: INFO: recursively deleting sub-directories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
        find ${DATA_DIR} -type d -name "${FLAG_DIR_SUFFIX}"          -exec rm -rf {} \; 
        echo "DO_ALL.SH: INFO: recursively deleting residual .tar files"
        find ${DATA_DIR} -type f -name "*.tar"                     -exec rm    {} \;
        echo "DO_ALL.SH: INFO: recursively deleting residual .gz  files"
        find ${DATA_DIR} -type f -name "*.gz"                      -exec rm    {} \;
        echo "DO_ALL.SH: INFO: recursively deleting '.fqln'            files created in earlier runs"
        find ${DATA_DIR} -type l -name "*.fqln"                    -exec rm    {} \;
        echo "DO_ALL.SH: INFO: recursively deleting 'entire_patch.npy' files created in earlier runs"
        find ${DATA_DIR} -type f -name "entire_patch.npy"          -exec rm    {} \; 
        echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
        find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} \;
        echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '*${RNA_FILE_REDUCED_SUFFIX}'"
        find ${DATA_DIR} -type f -name *${RNA_FILE_REDUCED_SUFFIX}          -exec rm -f {} \;
        echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
        find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -exec rm -f {} \;
        echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${MASK_FILE_NAME_SUFFIX}'"
        find ${DATA_DIR} -type f -name "${MASK_FILE_NAME_SUFFIX}"    -exec rm -f {} +
        echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
        find ${DATA_DIR} -type f -name "${RESIZED_FILE_NAME_SUFFIX}" -exec rm -f {} +
        echo "DO_ALL.SH: INFO: recursively deleting files (tiles)  matching this pattern:  '*.png'                            <<< for image mode, deleting all the .png files (i.e. tiles) can take quite some time as there can be up to hundreds of thousands"
        find ${DATA_DIR} -type f -name *.png                       -exec rm -f {} \;
    fi

    if [[ "$4" == "matched" ]]; 
      then
       echo "DO_ALL.SH: INFO: deleting all subdirectories of ${DATA_DIR} that do not contain BOTH an 'svs' file and a 'FKPM-UQ' file in accordance with 'matched' directive"
       for x in dataset/* ; do [[ -d $x ]] && ! [[ $( ls $x/*.txt *.svs 2> /dev/null | wc -l 2> /dev/null ) -eq 1 ]] && ( rm -rf $x ) ; done 
    fi
    
    tree ${DATA_DIR}
    cd ${BASE_DIR}
  
    echo "=====> STEP 2 OF 6: GENERATING TILES FROM SLIDE IMAGES"
    if [[ ${USE_TILER} == "external" ]] 
      then
        sleep ${SLEEP_TIME}
        ./start.sh
      else
        echo "DO_ALL.SH: INFO:  skipping external tile generation in accordance with user parameter 'USE_TILER'"
    fi
    
    if [[ ${INPUT_MODE} == "rna" ]] || [[ ${INPUT_MODE} == "image_rna" ]] ;
      then
        echo "=====> STEP 3 OF 6: REMOVING ROWS (RNA EXPRESSION DATA) FROM FPKM-UQ FILES WHICH DO NOT CORRESPOND TO TARGET GENE LIST"
        sleep ${SLEEP_TIME}
        cp $1_global/*of_interest ${DATA_DIR}
        cp $1_global/ENSG_UCSC_biomart_ENS_id_to_gene_name_table ${DATA_DIR}      
        python reduce_FPKM_UQ_files.py --data_dir ${DATA_DIR} --target_genes_reference_file ${TARGET_GENES_REFERENCE_FILE} --rna_file_suffix ${RNA_FILE_SUFFIX} --rna_file_reduced_suffix ${RNA_FILE_REDUCED_SUFFIX}  \
        --rna_exp_column ${RNA_EXP_COLUMN} --use_unfiltered_data ${USE_UNFILTERED_DATA}
        
        echo "=====> STEP 4 OF 6: EXTRACTING RNA EXPRESSION INFORMATION AND SAVING AS NUMPY FILES"
        sleep ${SLEEP_TIME}
        python process_rna_exp.py --data_dir ${DATA_DIR} --rna_file_suffix ${RNA_FILE_SUFFIX} --rna_file_reduced_suffix ${RNA_FILE_REDUCED_SUFFIX} --rna_exp_column ${RNA_EXP_COLUMN} --rna_numpy_filename ${RNA_NUMPY_FILENAME} \
        --use_unfiltered_data ${USE_UNFILTERED_DATA}
    fi
    
    echo "=====> STEP 5 OF 6: PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
    sleep ${SLEEP_TIME}
    cp $1_global/mapping_file                   ${DATA_DIR}
    cp $1_global/${ENSG_REFERENCE_FILE_NAME}    ${DATA_DIR}  
    python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  
    
    NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
    echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}
fi

echo "=====> STEP 6 OF 6: LAUNCHING THE MAIN APPLICATION"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --use_tiler ${USE_TILER} --just_profile ${JUST_PROFILE} --just_test ${JUST_TEST} --skip_tiling ${SKIP_TILING} --skip_generation ${SKIP_GENERATION} --dataset ${DATASET} --data_dir ${DATA_DIR} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--ddp ${DDP} --use_autoencoder_output ${USE_AUTOENCODER_OUTPUT} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  --use_unfiltered_data ${USE_UNFILTERED_DATA} --remove_low_expression_genes  ${REMOVE_LOW_EXPRESSION_GENES} \
--low_expression_threshold ${LOW_EXPRESSION_THRESHOLD} --remove_unexpressed_genes ${REMOVE_UNEXPRESSED_GENES} \
--a_d_use_cupy ${A_D_USE_CUPY} --cov_threshold ${COV_THRESHOLD} --cov_uq_threshold ${COV_UQ_THRESHOLD} --cutoff_percentile ${CUTOFF_PERCENTILE} \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --nn_mode ${NN_MODE} --use_same_seed ${USE_SAME_SEED} --nn_type_img ${NN_TYPE_IMG} --nn_type_rna ${NN_TYPE_RNA}  \
--nn_dense_dropout_1 ${NN_DENSE_DROPOUT_1} --nn_dense_dropout_2 ${NN_DENSE_DROPOUT_2} \
--encoder_activation ${ENCODER_ACTIVATION} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} \
--gene_data_norm ${GENE_DATA_NORM} --gene_data_transform ${GENE_DATA_TRANSFORM} --gene_embed_dim ${GENE_EMBED_DIM} --hidden_layer_neurons ${HIDDEN_LAYER_NEURONS} --hidden_layer_encoder_topology ${HIDDEN_LAYER_ENCODER_TOPOLOGY} \
--cancer_type ${CANCER_TYPE} --cancer_type_long ${CANCER_TYPE_LONG} --class_names ${CLASS_NAMES} --long_class_names ${LONG_CLASS_NAMES} --class_colours ${CLASS_COLOURS} \
--n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} \
--greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} --label_swap_perunit ${LABEL_SWAP_PERUNIT} \
--target_tile_offset ${TARGET_TILE_OFFSET} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}  --points_to_sample ${POINTS_TO_SAMPLE} \
--show_rows ${SHOW_ROWS} --show_cols ${SHOW_COLS} --figure_width ${FIGURE_WIDTH} --figure_height ${FIGURE_HEIGHT} --annotated_tiles ${ANNOTATED_TILES} --supergrid_size ${SUPERGRID_SIZE} \
--patch_points_to_sample ${PATCH_POINTS_TO_SAMPLE} --scattergram ${SCATTERGRAM} --show_patch_images ${SHOW_PATCH_IMAGES} \
--probs_matrix ${PROBS_MATRIX} --probs_matrix_interpolation ${PROBS_MATRIX_INTERPOLATION} 
cd ${BASE_DIR}


echo "===> FINISHED "
