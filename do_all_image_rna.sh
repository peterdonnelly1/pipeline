#!/bin/bash

# exit if any command fails
# set -e
set -x

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

BASE_DIR=/home/peter/git/pipeline                                                                          # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=working_data                                                                                      # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"

#~ source conf/variables.sh ${DATASET}

rm logs/model_image.pt                > /dev/null 2>&1                                                     # delete existing trained image model,       if same exists
rm classi/modes/classify/train.pth    > /dev/null 2>&1                                                     # delete existing         input dataset,     if same exists
./do_all.sh   -d stad  -i image                    -c UNIMODE_CASE  -v True                                # train image      model against unimode training cases      <<<< NOTE: -v ('divide_classes') option causes the cases to be divided into UNIMODE_CASE____MATCHED and MULTIMODE____TEST. Do this once only.
./do_all.sh   -d stad  -i image      -m image_rna  -c UNIMODE_CASE  -X True -s True -g True -j True        # test  image      model against held out unimode cases   
                                                                                                           #    -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)

rm logs/model_rna.pt                  > /dev/null 2>&1                                                     # delete existing trained rna seq model,     if same exists
rm classi/modes/classify/train.pth    > /dev/null 2>&1                                                     # delete existing         input   dataset,   if same exists
./do_all.sh   -d stad  -i rna                      -c UNIMODE_CASE                                         # train  rna seq   model against unimode training cases
./do_all.sh   -d stad  -i rna        -m image_rna  -c UNIMODE_CASE                     -j True             # test   rna seq   model against held out unimode cases 
                                                                                                           #    -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)


rm logs/model_image_rna.pt            > /dev/null 2>&1                                                     # delete existing trained multimode model,   if same exists
#~ rm classi/modes/classify/train.pth    > /dev/null 2>&1                                                     # delete existing         input     dataset, if same exists
#~ ./do_all.sh   -d stad  -i image_rna                -c UNIMODE_CASE                                         # train  multimode model, using concatenation of feature vectors generated in the image and rna seq test runs above as inputs

#~ # Don't do this run, even though it might seem like a good idea.
#~ # There is no way to ensure image_rna test embeddings would correspond 1:1 with their associated image embeddings, thus it's very likely high that such an image_rna embedding test set would be heavily polluted with image_rna training examples
#~ ./just_test.sh   -d stad  -i image_rna                -c UNIMODE_CASE____MATCHED  

#~ # At this point we have the three trained models. Now swap to the test cases that were reserved for dual mode image+rna using the flag 'MULTIMODE____TEST'


# Generate image embeddings using optimised model
#~ echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE}'"
#~ find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}           -delete                           # delete any existing image feature files
#~ ./do_all.sh   -d stad  -i image      -m image_rna  -c MULTIMODE____TEST                -j True             # generate image feature files from the (reserved) multimode test set using trained image model

# Generate rna embeddings using optimised model
#~ echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_RNA}'"
#~ find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}             -delete                           # delete any existing rna seq feature files
#~ ./do_all.sh   -d stad  -i rna        -m image_rna  -c MULTIMODE____TEST                -j True             # generate image feature files from the (reserved) multimode test set using trained rna seq model

# Process matched image+rna emmbeddings
#~ echo "DO_ALL.SH: INFO: recursively deleting files matching this pattern:  '${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}'"
#~ find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}       -delete                           # delete any existing multimode feature files
#~ ./do_all.sh   -d stad  -i image_rna  -m image_rna  -c MULTIMODE____TEST                -j True             # classify the (reserved) multimode test cases trained multimode model
