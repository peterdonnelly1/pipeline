#!/bin/bash
#set -e

MAGENTA='\e[38;2;255;0;255m'
ORANGE='\e[38;2;255;103;0m'
CYAN='\e[36;1m'
RED='\e[38;2;255;0;0m'
RESET='\e[m'


alias cls='printf "\033c"'
SLEEP_TIME=0

# main directory paths & file names

BASE_DIR=/home/peter/git/pipeline                                                                          # root directory for everything (shell scripts, code, datasets, logs ...)
APPLICATION_DIR=classi
DATA_ROOT=working_data                                                                                     # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_SOURCE=${BASE_DIR}/source_data/${DATASET}                                                             # structured directory containing dataset. A copy is made to DATA_ROOT. DATA_SOURCE is left untouched
DATA_DIR=${BASE_DIR}/${DATA_ROOT}                                                                          # location of the above. Not to be confused with DATA_SOURCE, which points to the master directory (via ${DATASET})
GLOBAL_DATA_DIR=${BASE_DIR}/global
GLOBAL_DATA=${GLOBAL_DATA_DIR}/${DATASET}_global                                                           # name of a custom mapping file, if one exists, else "none"
MAPPING_FILE_NAME=${DATASET}_mapping_file_MASTER.csv                                                       
MAPPING_FILE=${DATA_DIR}/${MAPPING_FILE_NAME}
LOG_DIR=${BASE_DIR}/logs

# common

SHOW_ROWS=1000
SHOW_COLS=100


# only used for image

RANDOM_TILES="True"                                                                                        # select tiles at random coordinates from image. Done AFTER other quality filtering
ANNOTATED_TILES="False"                                                                                    # show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
SCATTERGRAM="True"                                                                                         # show scattergram image in tensorboard
SHOW_PATCH_IMAGES="False"                                                                                  # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
PROBS_MATRIX="False"                                                                                       # supplement scattergram with a probabilities matrix image in tensorboard
PROBS_MATRIX_INTERPOLATION="spline16"                                                                      # interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
PATCH_POINTS_TO_SAMPLE=500                                                                                 # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
FIGURE_WIDTH=8
FIGURE_HEIGHT=8

# only used for rna

ENCODER_ACTIVATION="none"                                                                                  # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 


if [[ ${INPUT_MODE} == "image" ]]; then
  FINAL_TEST_BATCH_SIZE=2                                                                                  # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
else
  FINAL_TEST_BATCH_SIZE=141                                                                                # (rna mode doesn't need this because the entire batch can easily be accommodated)
fi

# 'pre-sets' for the five processing modes

if [[ ${MODE} == "classify" ]]; then
  #~ cp -f ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py_classify_version      ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py 
  MAIN_APPLICATION_NAME=classify.py                                                                        # use classify.py   for any "images + classes" dataset
  #~ DATASET_HELPER_APPLICATION_NAME=data.classify.generate_image                                             # use generate_images  for any "images + classes" dataset OTHER THAN MNIST
elif [[ ${MODE} == "pre_compress" ]]; then
  #~ cp -f ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py_pre_compress_version  ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py
  MAIN_APPLICATION_NAME=pre_compress.py                                                                    # use pre_compress.py   for pre-compressing a dataset
  #~ DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate                                               # use pre_compress      for pre-compressing a dataset
elif [[ ${MODE} == "analyse_data" ]]; then
  #~ cp -f ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py_analyse_data_version  ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py
  A_D_USE_CUPY='True'                                                                                      # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
  MAIN_APPLICATION_NAME=analyse_data.py                               
  #~ DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate           
elif [[ ${MODE} == "gtexv6" ]]; then  
  #~ cp -f ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py_gtexv6_version        ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py
  MAIN_APPLICATION_NAME=traindpcca.py                                                                      # use traindpcca.py    for the  MNIST "digit images + synthetic classes" dataset  
elif [[ ${MODE} == "mnist" ]]; then  
  SKIP_GENERATION="True"
  #~ cp -f ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py_mnist_version         ${BASE_DIR}/${APPLICATION_DIR}/modes/__init__.py
  MAIN_APPLICATION_NAME=traindpcca.py                                                                      # use traindpcca.py    for the  MNIST "digit images + synthetic classes" dataset  
  #~ DATASET_HELPER_APPLICATION_NAME=data.mnist.generate                                                   # use generate_mnist   for the  MNIST "digit images + synthetic classes" dataset
  #~ DATASET_HELPER_APPLICATION_NAME=data.classify.generate_image                                             # use generate_images  for any "images + classes" dataset OTHER THAN MNIST    
else
  echo "VARIABLES.SH: INFO: no such MODE as '${MODE}'"
  exit
fi




# other variabes used by shell scripts
SAVE_MODEL_NAME="model.pt"
LATENT_DIM=1
FLAG_DIR_SUFFIX="*_all_downloaded_ok"
MASK_FILE_NAME_SUFFIX="*_mask.png"
RESIZED_FILE_NAME_SUFFIX="*_resized.png"
RNA_FILE_SUFFIX="*FPKM-UQ.txt"
RNA_FILE_REDUCED_SUFFIX="_reduced"
RNA_NUMPY_FILENAME="rna.npy"
EMBEDDING_FILE_SUFFIX_RNA="___rna.npy"
EMBEDDING_FILE_SUFFIX_IMAGE="___image.npy"
EMBEDDING_FILE_SUFFIX_IMAGE_RNA="___image_rna.npy"
ENSG_REFERENCE_FILE_NAME='ENSG_reference'
ENS_ID_TO_GENE_NAME_TABLE='ENSG_UCSC_biomart_ENS_id_to_gene_name_table'
ENSG_REFERENCE_COLUMN=0
RNA_EXP_COLUMN=1                                                                                           # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)
CLASS_NUMPY_FILENAME="class.npy"
NAMES_COLUMN="type_s"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"


# dataset specific parameters

if [[ ${DATASET} == "stad" ]]; then

  # caution: Not all stad cases have both image and rna-seq samples. Further, for rna-seq, there are only five subtypes with meaningful numbers of rna-seq cases (hence HIGHEST_CLASS_NUMBER=4)

  CANCER_TYPE="STAD"
  CANCER_TYPE_LONG="Stomach_Adenocarcinoma"      
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"

elif [[ ${DATASET} == "coad" ]]; then
  
  CANCER_TYPE="COAD"
  CANCER_TYPE_LONG="Colon_Adenocarcinoma"      
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"

elif [[ ${DATASET} == "thym" ]]; then

  CANCER_TYPE="THYM"
  CANCER_TYPE_LONG="Thymoma"   
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"
  
elif [[ ${DATASET} == "dlbc" ]]; then

  CANCER_TYPE="DLBC"
  CANCER_TYPE_LONG="Diffuse_Large_B_Cell_Lymphoma"   
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"

elif [[ ${DATASET} == "sarc" ]]; then
  
  CANCER_TYPE="SARC"
  CANCER_TYPE_LONG="Sarcoma"   
  ENCODER_ACTIVATION="none"                                                                              # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"
  
elif [[ ${DATASET} == "kidn" ]]; then
  
  CANCER_TYPE="Kidney_Cancer"
  CANCER_TYPE_LONG="Kidney_Cancer"   
  ENCODER_ACTIVATION="none"                                                                                # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"

elif [[ ${DATASET} == "0008" ]]; then
  
  CANCER_TYPE="Pan_Cancer"
  CANCER_TYPE_LONG="Pan_Cancer"
  ENCODER_ACTIVATION="none"                                                                                # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
  STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
  TARGET_TILE_COORDS="5000 5500"





else
    echo "VARIABLES.SH: INFO: no such dataset as '${DATASET}'"
    exit
fi
