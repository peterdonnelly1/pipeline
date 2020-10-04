#!/bin/bash

# exit if any command fails
# set -e

source conf/variables.sh ${DATASET}

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

MAPPING_FILE="stad_global/nationwidechildrens.org_clinical_patient_stad.tsv"
    
echo "=====> STEP 1 OF 1 CREATING MASTER MAPPING_FILE"

python create_master_mapping_file.py  --base_dir ${BASE_DIR} ---data_dir ${DATA_DIR} --mapping_file ${MAPPING_FILE} --mapping_file_name ${MAPPING_FILE_NAME} --class_column ${CLASS_COLUMN} --case_column=${CASE_COLUMN}

echo "===> FINISHED "
