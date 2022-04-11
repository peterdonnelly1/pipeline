#!/bin/bash

# exit if any command fails
set -e
set -x

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

BASE_DIR=/home/peter/git/pipeline                                                                                    # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=working_data                                                                                               # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"

# defaults for use if user doesn't set an option

BATCH_SIZE=59
HIDDEN_LAYER_NEURONS=2100
LEARNING_RATE=.00001
N_EPOCHS=150
NN_DENSE_DROPOUT_1=.20
REGEN="False"
REPEAT=8

while getopts b:H:L:o:r:R:7: option                                                                                  # weird choice of letters is because they're the same as used in do_all.sh, where they are a few out of dozens of parameters
  do
    case "${option}"
    in
    b) BATCH_SIZE=${OPTARG};;              # not currently implemented but probably should                                                                  
    H) HIDDEN_LAYER_NEURONS=${OPTARG};;    # not currently implemented but probably should                                                                  
    L) LEARNING_RATE=${OPTARG};;           # not currently implemented but probably should                                                                
    o) N_EPOCHS=${OPTARG};;                # not currently implemented but probably should
    r) REGEN=${OPTARG};;                   # implemented
    R) REPEAT=${OPTARG};;                  # not currently implemented but probably should
    7) NN_DENSE_DROPOUT_1=${OPTARG};;      # not currently implemented but probably should
    esac
  done


rm logs/model_image.pt                              > /dev/null 2>&1                                                 #                          delete existing trained image model,                    if one exists
rm classi/modes/classify/dataset_image_train.pth    > /dev/null 2>&1                                                 #                          delete existing         pytorch input image dataset     if one exists
rm classi/modes/classify/dataset_image_test.pth     > /dev/null 2>&1                                                 #                          delete existing         pytorch input image  dataset    if one exists

#=========================================================================================================================================================
# The first 4 runs train an image-only model and an rna-only model, and from each generates the feature vectors used in Run 5 to train an image+rna model
#=========================================================================================================================================================

# Run 1
./do_all.sh   -d stad  -i image                    -c UNIMODE_CASE  -v True -o ${N_EPOCHS}  -r ${REGEN}              #                          train image      model against unimode training cases   <<<< NOTE: -v ('divide_classes') option causes the cases to be divided into UNIMODE_CASE____MATCHED and MULTIMODE____TEST. Do this once only.
# Output is a trained image model for use in Run 2 and Run 6
# Run 2
./do_all.sh   -d stad  -i image      -m image_rna  -c UNIMODE_CASE  -X True -s True -g True -j True                  # generate image features: using held out unimode image cases.  Skip pre-processing , tiling and dataset generation 
# Output is a set of image feature vectors, for use in Run 5 only                                                    #                          -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)

rm logs/model_rna.pt                                > /dev/null 2>&1                                                 #                          delete existing trained rna-seq model,                  if one exists
rm classi/modes/classify/dataset_rna_train.pth      > /dev/null 2>&1                                                 #                          delete existing         pytorch input rna   dataset     if one exists
rm classi/modes/classify/dataset_rna_test.pth       > /dev/null 2>&1                                                 #                          delete existing         pytorch input rna   dataset     if one exists

# Run 3
./do_all.sh   -d stad  -i rna                      -c UNIMODE_CASE                                                   #                          train  rna seq   model against unimode training cases
# Output is a trained rna  model for use in Run 4 and run 7 
# Run 4
./do_all.sh   -d stad  -i rna        -m image_rna  -c UNIMODE_CASE  -X True         -g True -j True                  # generate rna   features: using against held out unimode rna-seq cases. Skip pre-processing and dataset generation.
# Output is a set of rna   feature vectors, for use in Run 5 only                                                    #                          -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)


# the following datasets are no longer required; so deleted to avoid confusion
rm classi/modes/classify/dataset_image_train.pth   > /dev/null 2>&1
rm classi/modes/classify/dataset_image_test.pth    > /dev/null 2>&1
rm classi/modes/classify/dataset_rna_train.pth     > /dev/null 2>&1
rm classi/modes/classify/dataset_rna_test.pth      > /dev/null 2>&1



#=========================================================================================================================================================
# Run 5 - concatenates the feature vectors resulting from Runs 2 and 4 and uses these to train an image+rna (multimodal) model
#=========================================================================================================================================================

# delete this, if it already exists, as we are about to generate a new image+rna model
rm logs/model_image_rna.pt                         > /dev/null 2>&1


# Inputs are the feature vectors generated at Runs 2 and 4. These are concatenated in the 'generate' phase of Run 5
./do_all.sh   -d stad  -i image_rna                -c UNIMODE_CASE  -X True                                          #                          train  multimode model, using concatenation of feature vectors generated in the image and rna seq test runs above as inputs
# Output is a trained model for use in Run 8


#  we don't required the output of Run 5. In Run 8 we will use embeddings derived from the held out MULTIMODE____TEST dataset
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}           -delete                                    
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}             -delete
find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}       -delete   


# Don't do the following run, even though it might seem like a good idea.
# There is no way to ensure image_rna test embeddings would correspond 1:1 with their associated image embeddings, thus it's very likely high that such an image_rna embedding test set would be heavily polluted with image_rna training examples
#~ ./just_test.sh   -d stad  -i image_rna              -c UNIMODE_CASE


#=========================================================================================================================================================
# Now we have a trained image+rna model.  Runs 6&7 generates feature vectors from the held-out 'MULTIMODE____TEST' cases and Run 8 classifies them 
#=========================================================================================================================================================


#~ # Run 6 - generate image embeddings using the trained model from step 1 on the held-out (MULTIMODE____TEST) cases 
./do_all.sh   -d stad  -i image      -m image_rna  -c MULTIMODE____TEST   -X True -j True
# Output is a set of image feature vectors, for use in Run 8                                                         # -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)


#~ # Run 7 - generate rna   embeddings using the trained model from step 3 on the held-out (MULTIMODE____TEST) cases 
./do_all.sh   -d stad  -i rna        -m image_rna  -c MULTIMODE____TEST   -X True -j True
# Output is a set of image rna vectors, for use in Run 8                                                             # -m image_rna flag means (i) generate feature vectors for multimodal training (ii) use the TRAINING indices that were used for training (we want every feature vector we can get our hands on)


# Run 8 -   combines the embeddings from Runs 6 & 7 and classifies them using the trained image+rna model generated at step 5
./do_all.sh   -d stad  -i image_rna  -m image_rna  -c MULTIMODE____TEST   -X True -j True                          -b 10


