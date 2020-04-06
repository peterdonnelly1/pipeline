#!/bin/bash

# exit if any command fails
# set -e

source conf/variables.sh ${DATASET}

export MKL_DEBUG_CPU_TYPE=5

echo "===> STARTING"
if [[ "$3" == "regen" ]]; 
  then
    echo "=====> STEP 1 OF 6: REGENERATING DATASET FOLDER (THIS CAN TAKE UP TO SEVERAL MINUTES)"
    rm -rf ${DATA_DIR}
    rsync -ah --info=progress2 $1/ ${DATA_DIR}
  else
    echo "=====> STEP 1 OF 6: DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
    echo "DO_ALL.SH: INFO: deleting all empty subdirectories under                     '${DATA_DIR}'"
    find ${DATA_DIR} -type d -empty -delete
    echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
    echo "DO_ALL.SH: INFO: recursively deleting residual gz  files:                    '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type f -name "*.gz"                      -exec rm    {} \; 
    echo "DO_ALL.SH: INFO: recursively deleting residual tar files:                    '${FLAG_DIR_SUFFIX}'"
    find ${DATA_DIR} -type f -name "*.tar"                     -exec rm    {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RNA_NUMPY_FILENAME}'"
    find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '*${RNA_FILE_REDUCED_SUFFIX}'"
    find ${DATA_DIR} -type f -name *${RNA_FILE_REDUCED_SUFFIX}          -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
    find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -exec rm -f {} \;
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${MASK_FILE_NAME_SUFFIX}'"
    find ${DATA_DIR} -type f -name ${MASK_FILE_NAME_SUFFIX}    -exec rm -f {} +
    echo "DO_ALL.SH: INFO: recursively deleting files          matching this pattern:  '${RESIZED_FILE_NAME_SUFFIX}'"
    find ${DATA_DIR} -type f -name ${RESIZED_FILE_NAME_SUFFIX} -exec rm -f {} +
    echo "DO_ALL.SH: INFO: recursively deleting files (tiles)  matching this pattern:  '*.png'                            <<< this one can take some time in the case of image mode"
    find ${DATA_DIR} -type f -name *.png                       -exec rm -f {} \;
fi

echo "===> STARTING"
if [[ "$4" == "matched" ]]; 
  then
   echo "DO_ALL.SH: INFO: deleting all subdirectories of ${DATA_DIR} that do not contain BOTH an 'svs' file and a 'FKPM-UQ' file in accordance with 'matched' directive"
   for x in dataset/* ; do [[ -d $x ]] && ! [[ $( ls $x/*.txt *.svs 2> /dev/null | wc -l 2> /dev/null ) -eq 1 ]] && ( rm -rf $x ) ; done 
fi

tree ${DATA_DIR}

cd ${BASE_DIR}

echo "=====> STEP 2 OF 6: GENERATING TILES FROM SLIDE IMAGES"
if [[ ${USE_TILER} == "external" ]]; 
  then
    sleep ${SLEEP_TIME}
    ./start.sh
  else
    echo "DO_ALL.SH: INFO:  skipping external tile generation in accordance with user parameter 'USE_TILER'"
fi

if [[ ${INPUT_MODE} == "rna" ]];
  then
    echo "=====> STEP 3 OF 6: REMOVING ROWS (RNA EXPRESSION DATA) FROM FPKM-UQ FILES WHICH DO NOT CORRESPOND TO A PMCC GENE PANEL GENE"
    sleep ${SLEEP_TIME}
    cp $1_global/pmcc_genes_reference_file ${DATA_DIR};
    python reduce_FPKM_UQ_files.py "--data_dir="${DATA_DIR} "--pmcc_genes_reference_file="${PMCC_GENES_REFERENCE_FILE} "--rna_file_suffix="${RNA_FILE_SUFFIX} "--rna_file_reduced_suffix" ${RNA_FILE_REDUCED_SUFFIX}  "--rna_exp_column="${RNA_EXP_COLUMN}
    
    echo "=====> STEP 4 OF 6: EXTRACTING RNA EXPRESSION INFORMATION AND SAVING AS NUMPY FILES"
    sleep ${SLEEP_TIME}
    python process_rna_exp.py "--data_dir="${DATA_DIR} "--rna_file_reduced_suffix" ${RNA_FILE_REDUCED_SUFFIX} "--rna_exp_column="${RNA_EXP_COLUMN} "--rna_numpy_filename="${RNA_NUMPY_FILENAME}
  else
    echo "=====> STEP 3 OF 6: REMOVING ROWS (RNA EXPRESSION DATA) FROM FPKM-UQ FILES WHICH DO NOT CORRESPOND TO A PMCC GENE PANEL GENE"
    echo "DO_ALL.SH: INFO:  image mode selected so skipping rna processing steps"  
    echo "=====> STEP 4 OF 6: EXTRACTING RNA EXPRESSION INFORMATION AND SAVING AS NUMPY FILES"
    echo "DO_ALL.SH: INFO:  image mode selected so skipping rna processing steps"  
fi

echo "=====> STEP 5 OF 6: PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
sleep ${SLEEP_TIME}
cp $1_global/mapping_file              ${DATA_DIR};
python process_classes.py "--data_dir="${DATA_DIR} "--class_numpy_filename="${CLASS_NUMPY_FILENAME} "--mapping_file="${MAPPING_FILE} "--case_column="${CASE_COLUMN} "--class_column="${CLASS_COLUMN}  

if [[ ${INPUT_MODE} == "rna" ]];
  then 
    NUMBER_OF_TILES=$(find ${DATA_DIR} -name *${TILE_SIZE}.png | wc -l)
    echo "DO_ALL.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}
fi

echo "=====> STEP 6 OF 6: RUNNING THE NETWORK"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --use_tiler ${USE_TILER} --just_profile ${JUST_PROFILE} --just_test ${JUST_TEST} --skip_preprocessing 'False' --skip_generation 'False' --dataset ${DATASET} --data_dir ${DATA_DIR} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --nn_mode ${NN_MODE} --nn_type ${NN_TYPE} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} --n_genes ${N_GENES} --gene_data_norm ${GENE_DATA_NORM} \
--class_names ${CLASS_NAMES} --class_colours ${CLASS_COLOURS} --n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} --greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT} \
--target_tile_coords ${TARGET_TILE_COORDS} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}
cd ${BASE_DIR}


echo "===> FINISHED "
