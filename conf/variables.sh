#!/bin/bash
#set -e
alias cls='printf "\033c"'
SLEEP_TIME=0

# main directory paths
BASE_DIR=/home/peter/git/pipeline
DATA_ROOT=dataset
DATA_DIR=${BASE_DIR}/${DATA_ROOT}
LOG_DIR=${BASE_DIR}/logs
NN_APPLICATION_PATH=dpcca

NN_MODE="dlbcl_image"                                                    # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
NN_MODE="pre_compress"                                                    # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
#NN_MODE="analyse_data"                                                   # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
JUST_PROFILE="False"                                                      # if "True" just analyse slide/tiles then exit
JUST_TEST="False"                                                         # if "True" don't train, but rather load model from disk and run test batches through it

DATASET="$1"
INPUT_MODE="$2"

if [[ "$3" == "test" ]];                                                  # only 'dlbcl_image' mode is supported for test so might as well automatically select it
  then
    JUST_TEST="True"
    NN_MODE="dlbcl_image"
fi

if [[ ${NN_MODE} == "dlbcl_image" ]]                                      # at least for the time being, doing tiling and generation in 'dlbcl_image' mode because don't want to rejig the gtexv6 specific files to be able to do this
  then
    SKIP_PREPROCESSING="False"
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="False"       
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_dlbcl_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "pre_compress" ]]
    then
    SKIP_PREPROCESSING="False"                                             
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="True"                                            # if true, use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists                                            
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_pre_compress_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "analyse_data" ]]
    then
    SKIP_PREPROCESSING="False"                                             
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="True"                                            # if true, use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists                                            
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_analyse_data_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "gtexv6" ]]
    then  
    SKIP_PREPROCESSING="True"                                             # relies on data being separately pre-processed in dlbcl_image mode, as a preliminary step
    SKIP_GENERATION="True"                                                # relies on data being separately generated     in dlbcl_image mode, as a preliminary step
    USE_UNFILTERED_DATA="True"    
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_gtexv6_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "mnist" ]]
    then  
    SKIP_PREPROCESSING="True"                                             # relies on data being separately pre-processed in dlbcl_image mode, as a preliminary step
    SKIP_GENERATION="True"                                                # relies on data being separately generated     in dlbcl_image mode, as a preliminary step
    USE_UNFILTERED_DATA="False"      
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_mnist_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  else
    echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
fi


CLASS_COLOURS="darkorange       lime      olive      firebrick     dodgerblue    tomato     limegreen         darkcyan"
MAX_CONSECUTIVE_LOSSES=9999

USE_SAME_SEED="TRUE"                                                     # set to TRUE to use the same seed every time (Zero will be useds)

if [[ ${DATASET} == "stad" ]]; 
  then
  if [[ ${INPUT_MODE} == "image" ]] || [[ ${INPUT_MODE} == "image_rna" ]]; 
    then
      N_SAMPLES=199                                                      # on MOODUS 233 valid samples for STAD; on DREEDLE 229 valid samples for STAD
      N_EPOCHS=25
      #N_SAMPLES=50                                                      # 50 valid samples for STAD / rna and for MATCHED subset (images + rna)
      PCT_TEST=.3                                                        # proportion of samples to be held out for testing
      N_GENES=506                                                        # 60482 genes in total for STAD rna-sq data of which 506 map to PMCC gene panel genes
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest
      GENE_DATA_NORM="NONE"                                              # supported options are NONE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG2PLUS1"                                    # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1
      TILE_SIZE="128"                                                    # must be a multiple of 64 
      TILES_PER_IMAGE=50                                                 # Training mode only. <450 for Moodus 128x128 tiles. (this parameter is automatically calculated in 'just_test mode')
      SUPERGRID_SIZE=2                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      BATCH_SIZE="16"                                                    # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
#     NN_TYPE="VGG11"                                                    # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
      NN_TYPE="DCGANAE128"                                                
      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 1)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 2)
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      LEARNING_RATE=".0008"
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach Adenocarcinoma"      
      CLASS_NAMES="diffuse_adenocar                   NOS_adenocar        intest_adenocar_muc                        intest_adenocar_NOS              intest_adenocar_pap                         intest_adenocar_tub                       signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_papillary_type  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#      STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                          # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                  # Show scattergram          view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                            #   In scattergram          view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                 # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FIGURE_WIDTH=9
      FIGURE_HEIGHT=9
  elif [[ ${INPUT_MODE} == "rna" ]];
    then
      N_SAMPLES=475                                                       # Max 50 valid samples for STAD / image <-- AND THE MATCHED SUBSET (IMAGES+RNA-SEQ)
      N_EPOCHS=100
      BATCH_SIZE="64"                                                    # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
      PCT_TEST=.2                                                         # proportion of samples to be held out for testing
#      N_GENES=60483                                                      # 60483 genes in total for STAD rna-sq data (505 map to PMCC gene panel genes of interest)
#      N_GENES=506
#      N_GENES=3141                                                       # 200810 - N_GENES is no longer used - now determined from examinging rna files - remove at leisure                  
      TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_transcripts_of_interest
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
                                                                         # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="False"                                             # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
      TILE_SIZE="128"                                                    # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=100                                                # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      SUPERGRID_SIZE=1                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
#      NN_TYPE="DENSE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5
      NN_TYPE="AEDEEPDENSE"                                                 # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
      NN_TYPE="TTVAE AEDEEPDENSE"                                                    # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
#     NN_DENSE_DROPOUT_1="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 1)
#     NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_1="0.6"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      LEARNING_RATE=".1"
#      LEARNING_RATE=".1 .08 .03 .01 .008 .003 .001 .0008"
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach Adenocarcinoma"      
      CLASS_NAMES="diffuse_adenocar                   NOS_adenocar        intest_adenocar_muc                        intest_adenocar_NOS              intest_adenocar_pap                         intest_adenocar_tub                       signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_papillary_type  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                          # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                  # Show scattergram          view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                            #   In scattergram          view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                 # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=40
      FIGURE_HEIGHT=60
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
  fi
elif [[ ${DATASET} == "sarc" ]];
  then
  if [[ ${INPUT_MODE} == "image" ]]; 
    then
      N_SAMPLES=104                                                       # 101 samples on Dreedle (but use n_tiles=100)
      N_GENES=505
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
      echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
  fi
elif [[ ${DATASET} == "mnist" ]];
  then
    if [[ ${INPUT_MODE} == "image_rna" ]]; 
      then
        MAX_CONSECUTIVE_LOSSES=10
        N_SAMPLES=60000
        N_EPOCHS=100
        BATCH_SIZE=64
        PCT_TEST=.1    
    else
      echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
    fi
else
    echo "VARIABLES.SH: INFO: no such dataset as '${DATASET}'"
fi



SAVE_MODEL_NAME="model.pt"
SAVE_MODEL_EVERY=5

if [[ ${NN_MODE} == "dlbcl_image" ]];
  then
    NN_MAIN_APPLICATION_NAME=trainlenet5.py                               # use trainlenet5.py   for any "images + classes" dataset
    NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image    # use generate_images  for any "images + classes" dataset OTHER THAN MNIST
    LATENT_DIM=1
elif [[ ${NN_MODE} == "pre_compress" ]];
  then
    NN_MAIN_APPLICATION_NAME=pre_compress.py                               # use pre_compress.py   for pre-compressing a dataset
    NN_DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate          # use pre_compress      for pre-compressing a dataset
    LATENT_DIM=1
elif [[ ${NN_MODE} == "analyse_data" ]];
  then
    NN_MAIN_APPLICATION_NAME=analyse_data.py                               
    NN_DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate           
    LATENT_DIM=1  

elif [[ ${NN_MODE} == "gtexv6" ]];
  then 
    NN_MAIN_APPLICATION_NAME=traindpcca.py                                # use traindpcca.py    for dlbcl or eye in dpcca mode
    NN_DATASET_HELPER_APPLICATION_NAME=data.gtexv6.generate               # use gtexv6           for dlbcl or eye in dpcca mode
    LATENT_DIM=2    
elif [[ ${NN_MODE} == "mnist" ]];
  then
    NN_MAIN_APPLICATION_NAME=traindpcca.py                                # use traindpcca.py    for the  MNIST "digit images + synthetic classes" dataset  
#    NN_DATASET_HELPER_APPLICATION_NAME=data.mnist.generate               # use generate_mnist   for the  MNIST "digit images + synthetic classes" dataset
    NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image    # use generate_images  for any "images + classes" dataset OTHER THAN MNIST    
    LATENT_DIM=1    
else
    echo "VARIABLES.SH: INFO: no such NN_MODE as '${NN_MODE}'"
fi

                                                       
USE_TILER='internal'                                                    # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version
#TILE_SIZE=299                                                          # PGD 202019 - Inception v3 requires 299x299 inputs (or does it? Other sizes seem to work - are the images being padded or trucnated by pytorch?)

MAKE_GREY_PERUNIT=0.0                                                   # make this proportion of tiles greyscale. used in 'dataset.py'. Not related to MINIMUM_PERMITTED_GREYSCALE_RANGE

MINIMUM_PERMITTED_GREYSCALE_RANGE=60                                     # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MINIMUM_PERMITTED_UNIQUE_VALUES=100                                      # tile must have at least this many unique values or it will be assumed to be degenerate
MIN_TILE_SD=2                                                            # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=100                                                     # Used for determining/culling background tiles via 'min_tile_sd', how many points to sample on a tile when making determination

# other variabes used by shell scripts
FLAG_DIR_SUFFIX="*_all_downloaded_ok"
MASK_FILE_NAME_SUFFIX="*_mask.png"
RESIZED_FILE_NAME_SUFFIX="*_resized.png"
RNA_FILE_SUFFIX="*FPKM-UQ.txt"
RNA_FILE_REDUCED_SUFFIX="_reduced"
RNA_NUMPY_FILENAME="rna.npy"
ENSG_REFERENCE_FILE_NAME='ENSG_reference'
ENSG_REFERENCE_COLUMN=0
RNA_EXP_COLUMN=1                                                        # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)

MAPPING_FILE=${DATA_DIR}/mapping_file
CLASS_NUMPY_FILENAME="class.npy"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
