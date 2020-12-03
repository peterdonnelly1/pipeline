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


rm logs/model_image.pt                > /dev/null 2>&1
rm dpcca/data/dlbcl_image/train.pth   > /dev/null 2>&1   
./do_all.sh      -d stad  -i image                    -c DESIGNATED_UNIMODE_CASE_FLAG
./just_test.sh   -d stad  -i image      -m image_rna  -c DESIGNATED_UNIMODE_CASE_FLAG

rm logs/model_rna.pt                  > /dev/null 2>&1
rm dpcca/data/dlbcl_image/train.pth   > /dev/null 2>&1
./do_all.sh      -d stad  -i rna                    -c DESIGNATED_UNIMODE_CASE_FLAG
./just_test.sh   -d stad  -i rna      -m image_rna  -c DESIGNATED_UNIMODE_CASE_FLAG

rm logs/model_image_rna.pt            > /dev/null 2>&1
rm dpcca/data/dlbcl_image/train.pth   > /dev/null 2>&1
./do_all.sh      -d stad  -i image_rna                -c DESIGNATED_UNIMODE_CASE_FLAG

# Don't do this run, even though it seems like it would be a good idea.
# There is no way of ensure the image_rna test embeddings correspong 1:1 with the image test embeddings; thus the likelihood is very high that the image_rna embedding test set will be heavily polluted with image_rna training examples
#~ ./just_test.sh   -d stad  -i image_rna                -c DESIGNATED_UNIMODE_CASE_FLAG  

#~ # At this point we have the three trained models. Now swap to the test cases that were reserved for dual mode image+rna using the flag 'DESIGNATED_MULTIMODE_CASE_FLAG'

# Generate image embeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}           -delete
./just_test.sh   -d stad  -i image      -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG

# Generate rna embeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_RNA}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}             -delete
./just_test.sh   -d stad  -i rna        -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG

#~ # Process matched image+rna emmbeddings
#~ echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}'"
#~ find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}       -delete
#~ ./just_test.sh   -d stad  -i image_rna  -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG
