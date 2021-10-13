#!/bin/bash

# exit if any command fails
# set -e

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

BASE_DIR=/home/peter/git/pipeline                                        # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                        # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"

#~ source conf/variables.sh ${DATASET}

# Process matched image+rna emmbeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}       -delete
./just_test.sh   -d stad  -i image_rna  -m image_rna  -c MULTIMODE_CASE____TEST_FLAG

