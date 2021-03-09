#!/bin/bash

# exit if any command fails
# set -e

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

BASE_DIR=/home/peter/git/pipeline                                                                          # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                                                          # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"

find ${DATA_DIR} -type f -name HAS_MATCHED_IMAGE_RNA_FLAG                       -delete                               
echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'DESIGNATED_UNIMODE_CASE_FLAG'"
find ${DATA_DIR} -type f -name DESIGNATED_UNIMODE_CASE_FLAG                     -delete
echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'DESIGNATED_MULTIMODE_CASE_FLAG'"
find ${DATA_DIR} -type f -name DESIGNATED_MULTIMODE_CASE_FLAG                   -delete                    # it's critical that existing  MULTIMODE cases are deleted, otherwise the image mode run and the rna mode run won't choose the same cases
echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'NOT_A_MULTIMODE_CASE_FLAG'"
find ${DATA_DIR} -type f -name NOT_A_MULTIMODE_CASE_FLAG                        -delete                    # it's critical that existing  NON-MULTIMODE cases are deleted, otherwise the image mode run and the rna mode run won't choose the same cases


rm logs/model_image.pt                > /dev/null 2>&1
rm dpcca/data/dlbcl_image/train.pth   > /dev/null 2>&1
./do_all.sh      -d stad  -i image                    -c NOT_A_MULTIMODE_CASE_FLAG       -v True           # -v ('divide_classes') option causes the cases to be divided into DESIGNATED_UNIMODE_CASE_FLAG and DESIGNATED_MULTIMODE_CASE_FLAG. Do this once only.
./just_test.sh   -d stad  -i image                    -c DESIGNATED_MULTIMODE_CASE_FLAG


rm logs/model_rna.pt                  > /dev/null 2>&1
rm dpcca/data/dlbcl_image/train.pth   > /dev/null 2>&1
./do_all.sh      -d stad  -i rna                      -c NOT_A_MULTIMODE_CASE_FLAG         
./just_test.sh   -d stad  -i rna                      -c DESIGNATED_MULTIMODE_CASE_FLAG
