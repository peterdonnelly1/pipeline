#!/bin/bash
#set -e

alias cls='printf "\033c"'

SLEEP_TIME=1

NN_MODE="dlbcl_image"                                                    # "image" and "rna" are supported
JUST_PROFILE="False"                                                     # If "True" just analyse slide/tiles then exit
JUST_TEST='False'                                                        # If "True" don't train, but rather load model from disk and run test batches through it

DATASET="$1"
INPUT_MODE="$2"

CLASS_COLOURS="darkorange         yellow        khaki               rosybrown                                  deepskyblue                      tomato                                      gold         cyan"

if [[ ${DATASET} == "stad" ]]; 
  then
  if [[ ${INPUT_MODE} == "image" ]]; 
    then
      N_SAMPLES=2                                                         # on MOODUS 233 valid samples for STAD but use 232 / image; on DREEDLE 229 valid samples for STAD (but set N_SAMPLES=228)
      #N_SAMPLES=49                                                       # 49 valid samples for STAD / image <-- IN THE CASE OF THE MATCHED SUBSET
      PCT_TEST=.1                                                         # proportion of samples to be held out for testing
      N_GENES=60482
      GENE_DATA_NORM="NONE"                                               # supported options are NONE, GAUSSIAN
      SUPERGRID_SIZE=2                                                    # for test mode only: enables 'super-patches' that combinine multiple batches into a grid [test_mode (only). Minimum/default value=1; maximum value depends in TILES_PER_IMAGE
      TILE_SIZE="128"                                                     # 
      TILES_PER_IMAGE=64                                                 # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for test_mode)
      BATCH_SIZE="64"
      NN_TYPE="VGG11"                                                     # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5
      RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                 # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      N_EPOCHS=40
      LEARNING_RATE=.001
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach Adenocarcinoma"      
      CLASS_NAMES="diffuse_adenocar                   NOS_adenocar        intest_adenocar_muc                        intest_adenocar_NOS              intest_adenocar_pap                         intest_adenocar_tub                       signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_papillary_type  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#      STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
  elif [[ ${INPUT_MODE} == "rna" ]];
    then
      N_SAMPLES=49                                                        # 50 valid samples for STAD / rna (and STAD / matched)
      PCT_TEST=0.1                                                        # proportion of samples to be held out for testing
      N_GENES=506
      GENE_DATA_NORM="NONE GAUSSIAN"                                      # supported options are NONE, GAUSSIAN
      TILES_PER_IMAGE=200
      TILE_SIZE=256                                                       # PGD 200428
      BATCH_SIZE=10
      NN_TYPE="DENSE"                                                     # supported options are LENET5, VGG11, VGG13, VGG16, VGG19, DENSE, CONV1D, INCEPT3
      NN_OPTIMIZER="RMSPROP"                                              # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
      N_EPOCHS=1
      LEARNING_RATE=".002 .001 .0009"
      CLASS_NAMES="diffuse_adenocar                   NOS_adenocar        intest_adenocar_muc                        intest_adenocar_NOS              intest_adenocar_pap                         intest_adenocar_tub                       signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_papillary_type  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="2905cbd1-719b-46d9-b8af-8fe4927bc473/TCGA-FX-A2QS-11A-01-TSA.536F63AE-AD9F-4422-8AC3-4A1C6A57E8D8.svs"
      TARGET_TILE_COORDS="3200 3200"
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
  fi
elif [[ ${DATASET} == "sarc" ]];
  then
  if [[ ${INPUT_MODE} == "image" ]]; 
    then
      N_SAMPLES=104                                                       # 101 samples on Dreedle (but use n_tiles=100)
      N_GENES=506
      GENE_DATA_NORM="NONE"                                               # supported options are NONE, GAUSSIAN
      TILES_PER_IMAGE=200
      TILE_SIZE=256                                                       # PGD 200428
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
      GENE_DATA_NORM="NONE GAUSSIAN"                                      # supported options are NONE, GAUSSIAN
      TILES_PER_IMAGE=150
      TILE_SIZE=256                                                       # PGD 200428
      NN_TYPE="DENSE"                                                     # supported options are LENET5, VGG11, VGG13, VGG16, VGG19, DENSE, CONV1D, INCEPT3
      NN_OPTIMIZER="ADAM RMSPROP SGD"                                     # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      RANDOM_TILES="True"                                                 # Select tiles at random coordinates from image. Done AFTER other quality filtering
      N_EPOCHS=1000
      BATCH_SIZE=50
      LEARNING_RATE=".01 .007 .001 .0007 .0001"
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
LOG_DIR="${BASE_DIR}/logs"
SAVE_MODEL_NAME="model.pt"
SAVE_MODEL_EVERY=5

NN_APPLICATION_PATH=dpcca
#NN_MAIN_APPLICATION_NAME=traindpcca.py                                 # use traindpcca.py for dlbcl or eye in dpcca mode
#NN_DATASET_HELPER_APPLICATION_NAME=data.gtexv6.generate                # use gtexv6        for dlbcl or eye in dpcca mode

NN_MAIN_APPLICATION_NAME=trainlenet5.py                                 # use trainlenet5.py   for any "images + classes" dataset
NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image      # use generate_images  for any "images + classes" dataset OTHER THAN MNIST
#NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_mnist     # use generate_mnist   for any  MNIST "images + classes" dataset

LATENT_DIM=1                                                            # use 1 for image only (NOTE: WILL BE OVERWRITTEN BY ITERTOOLS)
#LATENT_DIM=2                                                           # use 2 for DPCCA
MAX_CONSECUTIVE_LOSSES=9999
                                                       
USE_TILER='internal'                                                    # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version
#TILE_SIZE=299                                                          # PGD 202019 - Inception v3 requires 299x299 inputs

MAKE_GREY_PERUNIT=0.0                                                   # make this proportion of tiles greyscale. used in 'dataset.py'. Not related to MINIMUM_PERMITTED_GREYSCALE_RANGE

MINIMUM_PERMITTED_GREYSCALE_RANGE=60                                    # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MINIMUM_PERMITTED_UNIQUE_VALUES=100                                      # tile must have at least this many unique values or it will be assumed to be degenerate
MIN_TILE_SD=2                                                            # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=100                                                     # In support of culling slides using 'min_tile_sd', how many points to sample on a tile when making determination

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
