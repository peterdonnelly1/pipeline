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

BASE_DIR=/home/peter/git/pipeline                                        # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                        # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"


# Generate image embeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}           -delete
./just_test.sh   -d stad  -i image      -m image_rna  -c MULTIMODE____TEST

# Generate rna embeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_RNA}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}             -delete
./just_test.sh   -d stad  -i rna        -m image_rna  -c MULTIMODE____TEST

# Process matched image+rna emmbeddings
echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}'"
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}       -delete
./just_test.sh   -d stad  -i image_rna  -m image_rna  -c MULTIMODE____TEST
