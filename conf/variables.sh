#!/bin/bash
#set -e

########################################################################################################################################################
#
# NOTES REGARDING PARAMETERS THAT ARE ALLOWED TO HAVE MORE THAN ONE VALUE
#
# More than one value can be specified for the following ...
#
#   COMMON parameters: 
#     N_SAMPLES, BATCH_SIZE, NN_OPTIMIZER, LEARNING_RATE, PCT_TEST, LABEL_SWAP_PCT, HIGHEST_CLASS_NUMBER, LABEL_SWAP_PCT
#
#   IMAGE parameters: 
#     NN_TYPE_IMG, TILE_SIZE, N_TILES, RANDOM_TILES, STAIN_NORM, JITTER, MAKE_GREY_PCT
#
#   RNA parameters: 
#     NN_TYPE_RNA, HIDDEN_LAYER_NEURONS, NN_DENSE_DROPOUT_1, NN_DENSE_DROPOUT_2, GENE_DATA_NORM, GENE_DATA_TRANSFORM, GENE_EMBED_DIM
#
# If more than one value is specified for any of these, an experiment 'job' will be created and run
# The job will comprise one run for every combination of the specified parameters (Cartesian product of the parameters)
#
#    - values must be quoted & separated by spaces (not commas)  E.g. "3000 3500 4000"
#    -  values must ALWAYS be put in quotes, even if there is only a single value
#
#############################################################################################################################################################
#
# NOTES REGARDING the parameter 'HIDDEN_LAYER_ENCODER_TOPOLOGY', which it specifies number of layers and number of neurons per layers
#
#    (a)  This parameter can only be used with the DEEPDENSE, AEDEEPDENSE and TTVAE models 
#    (b)  there can only be one specification of HIDDEN_LAYER_ENCODER_TOPOLOGY per job
#
#############################################################################################################################################################

alias cls='printf "\033c"'
SLEEP_TIME=0



# main directory paths & file names
NN_APPLICATION_PATH=dpcca
BASE_DIR=/home/peter/git/pipeline                                        # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                        # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}                                        # location of the above. Not to be confused with DATA_SOURCE, which points to the master directory (via ${DATASET})
DATA_SOURCE=${BASE_DIR}/${DATASET}                                       # structured directory containing dataset. A copy is made to DATA_ROOT. DATA_SOURCE is left untouched
GLOBAL_DATA=${BASE_DIR}/${DATASET}_global                                # name of a custom mapping file, if one exists, else "none"
MAPPING_FILE_NAME=${DATASET}_mapping_file_MASTER.csv                     # mapping file to use, if it's a special one. (Default "mapping_file" (no extension), doesn't have to be specified)
MAPPING_FILE=${DATA_DIR}/${MAPPING_FILE_NAME}
LOG_DIR=${BASE_DIR}/logs

MINIMUM_JOB_SIZE=2                                                       # Only do a box plot if the job has at least this many runs (otherwise it's a bit meaningless)
CASES_RESERVED_FOR_IMAGE_RNA=5                                           # number of cases to be reserved for image+rna testing. <<< HAS TO BE ABOVE ABOUT 5 FOR SOME REASON -- NO IDEA WHY ATM
USE_SAME_SEED="False"                                                     # set to TRUE to use the same seed every time for random numbers generation, for reproducability across runs (i.e. so that results can be more validly compared)
JUST_PROFILE="False"                                                     # if "True" just analyse slide/tiles then exit
DDP="False"                                                              # PRE_COMPRESS mode only: if "True", use PyTorch 'Distributed Data Parallel' to make use of multiple GPUs. (Works on single GPU machines, but is of no benefit and has additional overhead, so should be disabled)

MOMENTUM=0.8                                                             # for use by t-sne
                                                                         
CLASS_COLOURS="darkorange       lime      olive      firebrick     dodgerblue    tomato     limegreen         darkcyan"
COLOUR_MAP="tab10"                                                       # see 'https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html' for allowed COLOUR_MAPs (Pastel1', 'Pastel2', 'Accent', 'Dark2' etc.)
BAR_CHART_X_LABELS="case_id"                                             # if "case_id" use the case id as the x-axis label for bar charts, otherwise use integer sequence
BAR_CHART_SORT_HI_LO="False"                                             # Some less important bar charts will be suppressed if it is set to 'False'
BAR_CHART_SHOW_ALL="False"
RENDER_CLUSTERING="True"
BOX_PLOT="True"                                                          # If true, do a Seaborn box plot for the job (one box plot is generated per 'job', not per 'run')

MAX_CONSECUTIVE_LOSSES=5                                               # training will stop after this many consecutive losses, regardless of nthe value of N_EPOCHS

ZOOM_OUT_MAGS="1"                                                        # image only. magnifications (compared to baseline magnification) to be used when selecting areas for tiling, chosen according to the probabilities contained in ZOOM_OUT_PROB
ZOOM_OUT_PROB="1"                                                        # image only. Chosen for magnification according to these probabilities, which must add up to 1

# 'Pre-sets"

if [[ ${NN_MODE} == "dlbcl_image" ]]
  then
    #~ USE_UNFILTERED_DATA="True"       
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_dlbcl_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py 
elif [[ ${NN_MODE} == "pre_compress" ]]
    then
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_pre_compress_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
elif [[ ${NN_MODE} == "analyse_data" ]]
    then
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_analyse_data_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
    A_D_USE_CUPY='True'                                                                                    # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
elif [[ ${NN_MODE} == "gtexv6" ]]
    then  
    USE_UNFILTERED_DATA="True"    
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_gtexv6_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
elif [[ ${NN_MODE} == "mnist" ]]
    then  
    SKIP_GENERATION="True"
    USE_UNFILTERED_DATA="False"      
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_mnist_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
else
    echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
fi


if [[ ${DATASET} == "stad" ]]; 
  then
  if [[ ${INPUT_MODE} == "image" ]]
    then
      FINAL_TEST_BATCH_SIZE=2                                            # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
      RANDOM_TILES="True"                                                # select tiles at random coordinates from image. Done AFTER other quality filtering
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach_Adenocarcinoma"      
      CLASS_NAMES="diffuse   tubular   mucinous    signet_ring    papillary   tubular  stomach_NOS    intestinal_NOS       none"
      LONG_CLASS_NAMES="diffuse   tubular   mucinous    signet_ring    papillary   tubular  stomach_NOS    intestinal_NOS       none"
      #~ STAIN_NORMALIZATION="spcn"                                         # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"

      # Vizualization related
      ANNOTATED_TILES="False"                                             # Show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
      SCATTERGRAM="True"                                                 # Show scattergram image in tensorboard
      SHOW_PATCH_IMAGES="False"                                           # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="False"                                                # Supplement scattergram with a probabilities matrix image in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      PATCH_POINTS_TO_SAMPLE=500                                          # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
      FIGURE_WIDTH=8
      FIGURE_HEIGHT=8

      N_GENES=777                                                        # 60482 genes in total for STAD rna-sq data of which 506 map to PMCC gene panel genes
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="2000"                                        # only used for AEDENSE and DENSE at the moment
      SHOW_ROWS=1000
      SHOW_COLS=100


  elif [[ ${INPUT_MODE} == "rna" ]]  
    then                                                                  # Also works well  HIDDEN_LAYER_NEURONS="700"; NN_DENSE_DROPOUT_1="0.2" <<< TRY IT AGAIN
                                                                          # Also works well  HIDDEN_LAYER_NEURONS="250"; NN_DENSE_DROPOUT_1="0.2"  << BEST SO FAR?
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/just_hg38_protein_coding_genes 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest        # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      #~ GENE_EMBED_DIM="100"                                               # only used for AEDENSE at the moment
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach_Adenocarcinoma"      
      #~ CLASS_NAMES="C1  C2  C3  C4  C5 C6  C7"
      #~ LONG_CLASS_NAMES="C1  C2  C3  C4  C5  C6  C7"
      CLASS_NAMES="diffuse tubular mucinous intest_nos adeno_nos"
      LONG_CLASS_NAMES="diffuse tubular mucinous intest_nos adeno_nos"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12

      TILE_SIZE="128"                                                    # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=1234                                               # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      #~ STAIN_NORMALIZATION="NONE"                                         # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                            # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      

      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=141                                           # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

  elif [[ ${INPUT_MODE} == "image_rna" ]]  
    then                                                                 
      #~ GENE_EMBED_DIM="100"                                               # only used for AEDENSE at the moment
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach_Adenocarcinoma"      
      #~ CLASS_NAMES="C1  C2  C3  C4  C5 C6  C7"
      #~ LONG_CLASS_NAMES="C1  C2  C3  C4  C5  C6  C7"
      CLASS_NAMES="diffuse, tubular, mucinous, intest_nos, adeno_nos"
      LONG_CLASS_NAMES="diffuse, tubular, mucinous, intest_nos, adeno_nos"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                            # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=100                                          # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)



  else
      echo "VARIABLES.SH: INFO: no such input mode as ${INPUT_MODE}"
  fi  







elif [[ ${DATASET} == "coad" ]]; 
  then
  
    CANCER_TYPE="COAD"
    CANCER_TYPE_LONG="Colon_Adenocarcinoma"      
    CLASS_NAMES="colon_adenocarcinoma   colon_mucinous_adeno"
    LONG_CLASS_NAMES="colon_adenocarcinoma   colon_mucinous_adenocarcinoma"


  if [[ ${INPUT_MODE} == "image" ]]
  
    then
      FINAL_TEST_BATCH_SIZE=2                                            # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
      RANDOM_TILES="True"                                                # select tiles at random coordinates from image. Done AFTER other quality filtering
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"

      # Vizualization related
      ANNOTATED_TILES="False"                                             # Show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
      SCATTERGRAM="True"                                                 # Show scattergram image in tensorboard
      SHOW_PATCH_IMAGES="False"                                           # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="False"                                                # Supplement scattergram with a probabilities matrix image in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      PATCH_POINTS_TO_SAMPLE=500                                          # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
      FIGURE_WIDTH=8
      FIGURE_HEIGHT=8

      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      SHOW_ROWS=1000
      SHOW_COLS=100


  elif [[ ${INPUT_MODE} == "rna" ]]  
    then                                                                  # Also works well  HIDDEN_LAYER_NEURONS="700"; NN_DENSE_DROPOUT_1="0.2" <<< TRY IT AGAIN
                                                                          # Also works well  HIDDEN_LAYER_NEURONS="250"; NN_DENSE_DROPOUT_1="0.2"  << BEST SO FAR?
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/just_hg38_protein_coding_genes 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/COAD_genes_of_interest        # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      #~ GENE_EMBED_DIM="100"                                               # only used for AEDENSE at the moment

      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12

      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                            # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      

      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=141                                           # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

  elif [[ ${INPUT_MODE} == "image_rna" ]]  
    then                                                                 
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                            # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=100                                          # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)


  else
      echo "VARIABLES.SH: INFO: no such input mode as ${INPUT_MODE}"
  fi  



  
  
else
    echo "VARIABLES.SH: INFO: no such dataset as '${DATASET}'"
fi




# instructions for using the autoencoder front end

# 1 set NN_MODE="pre_compress"
#       set JUST_TEST="False"
#       select an autoencoder (can't go wrong with AEDENSE for example)
#     select preferred dimensionality reduction
#        if using AEDENSE ...
#            set selected preferred values via HIDDEN_LAYER_NEURONS and GENE_EMBED_DIM
#              HIDDEN_LAYER_NEURONS          sets the number of neurons in the (single) hidden later
#              GENE_EMBED_DIM                sets the number of dimensions (features) that each sample will be reduced to
#        if using AEDEEPDENSE or TTVAE ...
#            set selected preferred values via HIDDEN_LAYER_ENCODER_TOPOLOGY and GENE_EMBED_DIM
#              HIDDEN_LAYER_ENCODER_TOPOLOGY sets the number of neurons in each of the (arbitrary number of) hidden laters. There's no upper limit on the number of hidden layers, but the gpu will eventually run out of memoery and crash
#              GENE_EMBED_DIhttps://en.wikipedia.org/wiki/ANSI_escape_codeM                sets the number of dimensions (features) that each sample will be reduced to
#       run the autoencoder using ./just_run.sh or ./do_all.sh
#       perform at least 1000 epochs of training
#
#     as training proceeeds, the system will automatically save the latest/best model to a file  that will be used up in step 2
#
#
#  once training has completed ...
#
# 2 remain in pre_compress mode
#     set JUST_TEST="True"
#       select an encoder (can't go wrong with DENSE for example)
#     set BATCH_SIZE to be the same value as N_SAMPLES (e.g. "475")
#     run the autoencoder using ./just_run.sh or ./do_all.sh
#         set selected preferred values via HIDDEN_LAYER_ENCODER_TOPOLOGY and cfg.GENE_EMBED_DIM.  
#     observe the terminal output to ensure the dimensionality reduction was successful (i.e. little information lost compared to the original values)  
#         the final array displayed should be very largely green if the autoencoder has performed well
#           bright green indicates that the reconstructed output was within    1% of the input for that value (e.g. rna-seq value) << excellent
#           pale   green indicates that the reconstructed output was within    5% of the input for that value (e.g. rna-seq value) << ok       if there's also a lot of great deal of bright green     values
#           orange       indicates that the reconstructed output was within   25% of the input for that value (e.g. rna-seq value) << ok       if there is only a small number      of orange and gold values
#           gold         indicates that the reconstructed output was within   50% of the input  for that value (e.g. rna-seq value) << only ok if there is only a small number      of orange and gold values
#           blue         indicates that the reconstructed output was more   >100% away from the input          (e.g. rna-seq value) << only ok if there is only tiny number         of blue            values
#
#     the system will save the encoded (dimensionality reduced) features to a file  that will be used up in step 3.
#
# 3 change mode to NN_MODE="dlbcl_image"
#    set     USE_UNFILTERED_DATA="True"       
#    set     JUST_TEST="False"
#    set     USE_AUTOENCODER_OUTPUT="True"
#    set     BATCH_SIZE back to an appropriate value for training (e.g. 32 or 64) 
#
#     run using ./just_run.sh or ./do_all.sh
#      
#   USE_AUTOENCODER_OUTPUT="True" will cause the system will used the ae feature file saved at step 2 instead of the usual pre-processed (e.g. rna-seq) values

# for STAD:
# 200913 - BEST       ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" (67%, 69%, 77%, 76%); overall 72.4% (results more consistent)
# 200913 - BEST       --->                                                                                                          HIDDEN_LAYER_NEURONS="1500";                                                                             overall 72.1%
# 200913 - OK         --->                                                                                                          HIDDEN_LAYER_NEURONS="1400";                                                                             overall 72.0%
# 200913 - OK         --->                                                                                                          HIDDEN_LAYER_NEURONS="1250";                                                                             overall 72.1%
# 200913 - OK         ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" (58%, 68%, 71%, 73%); overall 67.7%
# 200913 - Works well ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.3; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE  (52%, 64%, 74%  75%); overall 68.8%
# 200913 - OK         ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="32"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" (59%, 66%, 71%, 74%); overall 67.4%
# 200913 - OK         ---> USE_SAME_SEED="False"; N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" best batch was 80.21%)
# 200913 - Average    ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="700";  NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" (67%, 69%, 77%, 76%); overall 69%
# 200913 - Poor       ---> USE_SAME_SEED="True";  N_EPOCHS=200; N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="200";  NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" (59%, 65%, 66%, 68%); overall 65%



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

                                                       
USE_TILER='internal'                                                     # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version

MINIMUM_PERMITTED_GREYSCALE_RANGE=150                                     # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MINIMUM_PERMITTED_UNIQUE_VALUES=150                                      # tile must have at least this many unique values or it will be assumed to be degenerate
MIN_TILE_SD=2                                                            # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=100                                                     # Used for determining/culling background tiles via 'min_tile_sd', how many points to sample on a tile when making determination


# other variabes used by shell scripts
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
ENSG_REFERENCE_COLUMN=0
RNA_EXP_COLUMN=1                                                        # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)
CLASS_NUMPY_FILENAME="class.npy"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
