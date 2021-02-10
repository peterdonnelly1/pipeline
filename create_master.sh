#!/bin/bash

# exit if any command fails
# set -e


export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

    
echo "=====> STEP 1 OF 1 CREATING MASTER MAPPING_FILE"

python create_master_mapping_file.py  --dataset $1
echo "===> FINISHED "
