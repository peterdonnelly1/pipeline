#!/bin/bash

# exit if any command fails
# set -e


#export MKL_DEBUG_CPU_TYPE=5   # ONLY USE WITH INTEL CPUS (NOT AMD)
#export KMP_WARNINGS=FALSE     # ONLY USE WITH INTEL CPUS (NOT AMD)

BASE_DIR=/home/peter/git/pipeline                                                                          # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_SOURCE=${BASE_DIR}/source_data/${DATASET}                                                             # structured directory containing dataset. A copy is made to DATA_ROOT. DATA_SOURCE is left untouched
DATA_DIR=${BASE_DIR}/${DATA_ROOT}                                                                          # location of the above. Not to be confused with DATA_SOURCE, which points to the master directory (via ${DATASET})
GLOBAL_DATA_DIR=${BASE_DIR}/global
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
    
echo "=====> STEP 1 OF 1 CREATING MASTER MAPPING_FILE"

python create_master_mapping_file.py  --dataset $1 --data_source ${DATA_SOURCE}  --global_data_dir ${GLOBAL_DATA_DIR} --case_column ${CASE_COLUMN} --class_column ${CLASS_COLUMN}
echo "===> FINISHED "
