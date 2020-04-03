#!/bin/bash
#set -e

alias cls='printf "\033c"'

SLEEP_TIME=1

NN_MODE="dlbcl_image"                                                        # only "image" and "rna" are supported
JUST_PROFILE="False"                                                     # If "true" just analyse slide/tiles then exit

DATASET="$1"
INPUT_MODE="$2"

if [[ ${DATASET} == "stad" ]]; 
  then
    N_SAMPLES=227                                                       # 231 valid samples for STAD
    N_GENES=60482
    TILES_PER_IMAGE="100"
    NN_TYPE="VGG11"                                                     # supported options are VGG11, VGG13, VGG16, VGG19 
    RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
    NN_OPTIMIZER="ADAM"                                                 # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
    N_EPOCHS=150
    BATCH_SIZE=65
    LEARNING_RATE=.00015
    CLASS_NAMES="diffuse_adenocar NOS_adenocar  intest_adenocar_muc  intest_adenocar_NOS  intest_adenocar_pap  intest_adenocar_tub  signet_ring"
    STAIN_NORMALIZATION="NONE"                            # options are NONE, reinhard, spcn  (used in 'save_svs_to_tiles' to specify the type of colour normalization to be performed)
    STAIN_NORM_TARGET="be6531b2-d1f3-44ab-9c02-1ceae51ef2bb/TCGA-3M-AB46-01Z-00-DX1.70F638A0-BDCB-4BDE-BBFE-6D78A1A08C5B.svs"
    TARGET_TILE_COORDS="5000 5500"
elif [[ ${DATASET} == "sarc" ]];
  then
  if [[ ${INPUT_MODE} == "image" ]]; 
    then
      N_SAMPLES=104
      N_GENES=506
      TILES_PER_IMAGE=200
      NN_TYPE="VGG11"                                                     # supported options are VGG11, VGG13, VGG16, VGG19
      NN_OPTIMIZER="ADAM"                                                 # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
      N_EPOCHS=1000
      BATCH_SIZE=64
      LEARNING_RATE=.00083
      CLASS_NAMES="dediff_liposarcoma leiomyosarcoma myxofibrosarcoma pleomorphic_MFH synovial undiff_pleomorphic MPNST desmoid giant_cell_MFH"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (used in 'save_svs_to_tiles' to specify the type of colour normalization to be performed)
      STAIN_NORM_TARGET="2905cbd1-719b-46d9-b8af-8fe4927bc473/TCGA-FX-A2QS-11A-01-TSA.536F63AE-AD9F-4422-8AC3-4A1C6A57E8D8.svs"
      TARGET_TILE_COORDS="3200 3200"
  elif [[ ${INPUT_MODE} == "rna" ]];
    then
      N_SAMPLES=104
      N_GENES=506
      TILES_PER_IMAGE=200
      NN_TYPE="DENSE"                                                     # supported options are LENET5, VGG11, VGG13, VGG16, VGG19, DENSE, CONV1D, INCEPT3
      NN_OPTIMIZER="ADAM RMSPROP SGD ADADELTA"                                         # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
      N_EPOCHS=1000
      BATCH_SIZE="50 40 30 20 10"
      LEARNING_RATE=.0004
      CLASS_NAMES="dediff_liposarcoma leiomyosarcoma myxofibrosarcoma pleomorphic_MFH synovial undiff_pleomorphic MPNST desmoid giant_cell_MFH"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (used in 'save_svs_to_tiles' to specify the type of colour normalization to be performed)
      STAIN_NORM_TARGET="2905cbd1-719b-46d9-b8af-8fe4927bc473/TCGA-FX-A2QS-11A-01-TSA.536F63AE-AD9F-4422-8AC3-4A1C6A57E8D8.svs"
      TARGET_TILE_COORDS="3200 3200"
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
  fi
else
    echo "VARIABLES.SH: INFO: no such dataset '${INPUT_MODE}'"
fi

# main directory paths
BASE_DIR=/home/peter/git/pipeline
DATA_ROOT=dataset
DATA_DIR=${BASE_DIR}/${DATA_ROOT}

NN_APPLICATION_PATH=dpcca
#NN_MAIN_APPLICATION_NAME=traindpcca.py                                 # use traindpcca.py for dlbcl or eye in dpcca mode
#NN_DATASET_HELPER_APPLICATION_NAME=data.gtexv6.generate                # use gtexv6        for dlbcl or eye in dpcca mode

NN_MAIN_APPLICATION_NAME=trainlenet5.py                                 # use trainlenet5.py   for any "images + classes" dataset
NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image      # use generate_images  for any "images + classes" dataset OTHER THAN MNIST
#NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_mnist     # use generate_mnist   for any  MNIST "images + classes" dataset

LATENT_DIM=1                                                            # use 1 for image only (NOTE: WILL BE OVERWRITTEN BY ITERTOOLS)
#LATENT_DIM=2                                                           # use 2 for DPCCA
MAX_CONSECUTIVE_LOSSES=9999
                                                       
TILE_SIZE=128                                                           # PGD 200108 - correct for gtexv6 experiment. It does not work with any old tile size, so be careful
USE_TILER='internal'                                                    # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version
#TILE_SIZE=299                                                          # PGD 202019 - Inception v3 requires 299x299 inputs

MINIMUM_PERMITTED_GREYSCALE_RANGE=0                                    # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MAKE_GREY_PERUNIT=0.0                                                   # make this proportion of tiles greyscale. used in 'dataset.py'. Not related to MINIMUM_PERMITTED_GREYSCALE_RANGE
MINIMUM_PERMITTED_UNIQUE_VALUES=30                                     # tile must have at least this many unique values or it will be assumed to be degenerate
MIN_TILE_SD=2                                                           # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=30                                                    # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination

# other variabes used by shell scripts
FLAG_DIR_SUFFIX="*_all_downloaded_ok"
MASK_FILE_NAME_SUFFIX="*_mask.png"
RESIZED_FILE_NAME_SUFFIX="*_resized.png"
RNA_FILE_SUFFIX="*FPKM-UQ.txt"
RNA_FILE_REDUCED_SUFFIX="_reduced"
RNA_NUMPY_FILENAME="rna.npy"
RNA_EXP_COLUMN=1                                                        # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)

MAPPING_FILE=${DATA_DIR}/mapping_file
PMCC_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_genes_reference_file
CLASS_NUMPY_FILENAME="class.npy"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
