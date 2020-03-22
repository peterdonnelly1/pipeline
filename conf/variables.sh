#!/bin/bash

alias cls='printf "\033c"'

SLEEP_TIME=1

#DATASET="SARC"
DATASET="STAD"
NN_MODE="dlbcl_image"
NN_TYPE="VGG11"                                                         # default. Can change this in main loop of trainlenet5
INPUT_MODE="image"                                                      # only "image" and "rna" are supported

N_SAMPLES=100                                                           # for SARC
N_GENES=60482                                                           # for SARC

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

N_EPOCHS=100
BATCH_SIZE=64                                                           # NOTE: WILL BE OVERWRITTEN BY ITERTOOLS
LATENT_DIM=1                                                            # use 1 for image only (NOTE: WILL BE OVERWRITTEN BY ITERTOOLS)
#LATENT_DIM=2                                                           # use 2 for DPCCA
MAX_CONSECUTIVE_LOSSES=9999
                                                       
TILES_PER_IMAGE=100
TILE_SIZE=128                                                           # PGD 200108 - correct for gtexv6 experiment. It does not work with any old tile size, so be careful
USE_TILER='internal'                                                    # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version
RANDOM_TILES='True'                                                     # PGD 200312 - select tiles at random coordinates from image. Done AFTER other quality filtering
#TILE_SIZE=299                                                          # PGD 202019 - Inception v3 requires 299x299 inputs
MINIMUM_PERMITTED_GREYSCALE_RANGE=60                                    # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MAKE_GREY_PERUNIT=0.0                                                   # make this proportion of tiles greyscale. used in 'dataset.py'. Not related to MINIMUM_PERMITTED_GREYSCALE_RANGE
MINIMUM_PERMITTED_UNIQUE_VALUES=100                                     # tile must have at least this many unique values or it will be assumed to be degenerate
STAIN_NORMALIZATION="NONE"                                              # options are "NONE", "reinhard", "spcn"  (used in 'save_svs_to_tiles' to specify the type of colour normalization to be performed)
#STAIN_NORM_TARGET="2905cbd1-719b-46d9-b8af-8fe4927bc473/TCGA-FX-A2QS-11A-01-TSA.536F63AE-AD9F-4422-8AC3-4A1C6A57E8D8.svs"  # use for SARC
STAIN_NORM_TARGET="be6531b2-d1f3-44ab-9c02-1ceae51ef2bb/TCGA-3M-AB46-01Z-00-DX1.70F638A0-BDCB-4BDE-BBFE-6D78A1A08C5B.svs"  # use for STAD
MIN_TILE_SD=4                                                           # Used to cull slides with a very reduced greyscale palette such as background tiles

MIN_TILE_SD=4                                                           # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=100                                                    # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination

# other variabes used by shells scripts
FLAG_DIR_SUFFIX="*_all_downloaded_ok"
MASK_FILE_NAME_SUFFIX="*_mask.png"
RESIZED_FILE_NAME_SUFFIX="*_resized.png"
RNA_FILE_SUFFIX="*FPKM-UQ.txt"
RNA_NUMPY_FILENAME="rna.npy"
RNA_EXP_COLUMN=1                                                        # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)

MAPPING_FILE=${DATA_DIR}/mapping_file
CLASS_NUMPY_FILENAME=class.npy
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
