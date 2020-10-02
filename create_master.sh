#!/bin/bash

# exit if any command fails
# set -e

source conf/variables.sh ${DATASET}

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

    
echo "=====> STEP 1 OF 1 CREATING MASTER MAPPING_FILE"

python create_master_mapping_file.py  --data_dir ${DATA_DIR} --dataset ${DATASET} --global_data ${GLOBAL_DATA} --mapping_file ${MAPPING_FILE} --mapping_file_name ${MAPPING_FILE_NAME} --class_column ${CLASS_COLUMN} --case_column=${CASE_COLUMN}

echo "===> FINISHED "
