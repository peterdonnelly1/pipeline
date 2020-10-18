#!/bin/bash
#set -e
alias cls='printf "\033c"'
SLEEP_TIME=0

USE_SAME_SEED="True"                                                     # set to TRUE to use the same seed every time for random numbers generation, for reproducability across runs (i.e. so that results can be more validly compared)

#DATASET="$1"                                                             # e.g. stad, tcl, dlbcl, thym ...
#INPUT_MODE="$2"

# main directory paths & file names
NN_APPLICATION_PATH=dpcca
BASE_DIR=/home/peter/git/pipeline                                        # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                        # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}                                        # location of the above. Not to be confused with DATA_SOURCE, which points to the master directory (via ${DATASET})
DATA_SOURCE=${BASE_DIR}/${DATASET}                                       # structured directory containing dataset. A copy is made to DATA_ROOT. DATA_SOURCE is left untouched
GLOBAL_DATA=${BASE_DIR}/${DATASET}_global                                # name of a custom mapping file, if one exists, else "none"
MAPPING_FILE_NAME=${DATASET}_mapping_file_MASTER                                         # mapping file to use, if it's a special one. (Default "mapping_file" (no extension), doesn't have to be specified)
MAPPING_FILE=${DATA_DIR}/${MAPPING_FILE_NAME}
LOG_DIR=${BASE_DIR}/logs

NN_MODE="dlbcl_image"                                                    # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
#NN_MODE="pre_compress"                                                    # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
#NN_MODE="analyse_data"                                                   # supported modes are:'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
JUST_PROFILE="False"                                                      # if "True" just analyse slide/tiles then exit
JUST_TEST="False"                                                         # if "True" don't train at all, but rather load saved model and run test batches through it
DDP="False"                                                               # PRE_COMPRESS mode only: if "True", use PyTorch 'Distributed Data Parallel' to make use of multiple GPUs. (Works on single GPU machines, but is of no benefit and has additional overhead, so should be disabled)

USE_AUTOENCODER_OUTPUT="False"                                            # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values)   
BOX_PLOT="True"                                                           # If true, do a Seaborn box plot for the job (one box plot is generated per 'job', not per 'run')
MINIMUM_JOB_SIZE=5                                                       # Only do a box plot if the job has at least this many runs

if [[ ${JUST_TEST} == "test" ]];                                                  # only 'dlbcl_image' mode is supported for test so might as well automatically select it
  then
    JUST_TEST="True"
    NN_MODE="dlbcl_image"
  else
    JUST_TEST="False"
  
fi

if [[ ${NN_MODE} == "dlbcl_image" ]]                                     # at least for the time being, doing tiling and generation in 'dlbcl_image' mode because don't want to rejig the gtexv6 specific files to be able to do this
  then
    SKIP_TILING="False"
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="True"       
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_dlbcl_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "pre_compress" ]]
    then
    SKIP_TILING="False"                                             
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="True"                                           # if true, use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists                                            
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_pre_compress_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "analyse_data" ]]
    then
    SKIP_TILING="False"                                             
    SKIP_GENERATION="False"
    USE_UNFILTERED_DATA="True"                                           # if true, use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists                                            
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_analyse_data_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "gtexv6" ]]
    then  
    SKIP_TILING="True"                                                   # relies on data being separately pre-processed in dlbcl_image mode, as a preliminary step
    SKIP_GENERATION="True"                                               # relies on data being separately generated     in dlbcl_image mode, as a preliminary step
    USE_UNFILTERED_DATA="True"    
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_gtexv6_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  elif [[ ${NN_MODE} == "mnist" ]]
    then  
    SKIP_TILING="True"                                                    # relies on data being separately pre-processed in dlbcl_image mode, as a preliminary step
    SKIP_GENERATION="True"                                                # relies on data being separately generated     in dlbcl_image mode, as a preliminary step
    USE_UNFILTERED_DATA="False"      
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_mnist_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py   # silly way of doing this, but better than doing it manually every time
  else
    echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
fi

CLASS_COLOURS="darkorange       lime      olive      firebrick     dodgerblue    tomato     limegreen         darkcyan"
MAX_CONSECUTIVE_LOSSES=9999

# NOTES REGARDING PARAMATERS WHICH ARE ALLOWED TO HAVE MORE THAN ONE VALUE
#
# If more than one value is specified for more than one parameter, then:
#    (i)   the experiment job will comprise one run for each and every combination of the specified parameters (Cartesian product)
#    (ii)  the values must be quoted & separated by spaces (not commas).  E.g. "3000 3500 4000"
#    (iii) these parameters should ALWAYS be put in quotes, even if there is only a single value
#
# More than one value can be specified for the following
#   COMMON parameters: 
#     N_SAMPLES, BATCH_SIZE, NN_OPTIMIZER, LEARNING_RATE, LABEL_SWAP_PERUNIT
#
#   IMAGE parameters: 
#     NN_TYPE_IMG, TILE_SIZE, TILES_PER_IMAGE, RANDOM_TILES, STAIN_NORMALIZATION, MAKE_GREY_PERUNIT
#
#   RNA parameters: 
#     NN_TYPE_RNA, HIDDEN_LAYER_NEURONS, NN_DENSE_DROPOUT_1, NN_DENSE_DROPOUT_2, GENE_DATA_NORM, GENE_DATA_TRANSFORM, GENE_EMBED_DIM
#
#
# HIDDEN_LAYER_ENCODER_TOPOLOGY
#    (i)  for AEDEEPDENSE and TTVAE models only
#    (i)  specifies the number of layers and number of neurons per layers
#    (ii) can only be one specification of HIDDEN_LAYER_ENCODER_TOPOLOGY per jobs


if [[ ${DATASET} == "stad" ]]; 
  then
  if [[ ${INPUT_MODE} == "image" ]] || [[ ${INPUT_MODE} == "image_rna" ]]
    then
      N_SAMPLES="20"                                                      # 228 image files for STAD; 479 rna-seq samples (474 cases); 229 have both (a small number of cases have two rna-seq samples)
      N_EPOCHS=30                                                         # ignored in test mode
      BATCH_SIZE="64"                                                     # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
      PCT_TEST=".2"                                                      # proportion of samples to be held out for testing
      FINAL_TEST_BATCH_SIZE=5000                                          # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)
      TILE_SIZE="64"                                                     # must be a multiple of 64 
      TILES_PER_IMAGE="100"                                              # Training mode only. <450 for Moodus 128x128 tiles. (this parameter is automatically calculated in 'just_test mode')
      SUPERGRID_SIZE=2                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
     NN_TYPE_IMG="VGG11"                                               # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
#      NN_TYPE_IMG="AE3LAYERCONV2D"                                       # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
#     NN_TYPE_IMG="DCGANAE128"                                                
      RANDOM_TILES="True"                                                # select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      LEARNING_RATE=".001"
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach_Intestine_Adenocarcinoma"      
      CLASS_NAMES="diffuse                            stomach_NOS                 mucinous                                   intestinal_NOS                   tubular                     signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            other        mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  other       intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                         # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # ..in scattergram view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FIGURE_WIDTH=9
      FIGURE_HEIGHT=9

      NN_TYPE_RNA="DENSE"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 1)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 2)
      N_GENES=506                                                        # 60482 genes in total for STAD rna-sq data of which 506 map to PMCC gene panel genes
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest
      GENE_DATA_NORM="NONE"                                              # supported options are NONE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG2PLUS1"                                    # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="False"                                             # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000"                               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="2000"                                        # only used for AEDENSE and DENSE at the moment
      GENE_EMBED_DIM="2000"                                              # only used for AEDENSE at the moment
      SHOW_ROWS=1000
      SHOW_COLS=100

      
# instructions for using the autoencoder front end:

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


  elif [[ ${INPUT_MODE} == "rna" ]] || [[ ${INPUT_MODE} == "image_rna" ]]   
    then                                                                  # Also works well  HIDDEN_LAYER_NEURONS="700"; NN_DENSE_DROPOUT_1="0.2" <<< TRY IT AGAIN
                                                                          # Also works well  HIDDEN_LAYER_NEURONS="250"; NN_DENSE_DROPOUT_1="0.2"  << BEST SO FAR?
      N_SAMPLES="469"                                                       # 469 rna-seq samples (474 cases); 229 have both (a small number of cases have two rna-seq samples)
      N_EPOCHS=300
      BATCH_SIZE="32"                                                     #  number of samples in each "mini batch"
#      BATCH_SIZE="95 95 95 95 95 95 95 95 95"
      PCT_TEST="0.2"                                                      # proportion of samples to be held out for testing
#      LEARNING_RATE=".0008"
      LEARNING_RATE=".0002"                                               # learning rate for back propagation
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_transcripts_of_interest  # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest        # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
                                                                         # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="True"                                              # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
#      NN_TYPE_RNA="AELINEAR"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDEEPDENSE"                                             # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="TTVAE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDENSE"                                                 # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
       NN_TYPE_RNA="DENSE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      HIDDEN_LAYER_ENCODER_TOPOLOGY="7000 6000 6000 6000"               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000 8000"                          # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="1500"                                        # only used for AEDENSE and DENSE at the moment. 1100 is best for DENSE (not necessarily same for AEDENSE)
      GENE_EMBED_DIM="500"                                              # only used for AEDENSE at the moment
      NN_DENSE_DROPOUT_1="0.2"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 1)
#     NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
#      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      CANCER_TYPE="STAD"
      CANCER_TYPE_LONG="Stomach_Intestine_Adenocarcinoma"      
      CLASS_NAMES="diffuse                            stomach_NOS                 mucinous                                   intestinal_NOS                   tubular                     signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            other        mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  other       intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=40
      FIGURE_HEIGHT=60

      NN_TYPE_IMG="VGG11"                                                    # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
      TILE_SIZE="128"                                                    # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=100                                                # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      SUPERGRID_SIZE=1                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                          # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                  # Show scattergram          view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                            #   In scattergram          view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                 # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=5000                                         # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

            
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
  fi
fi
  
if [[ ${DATASET} == "thym" ]]; 
  then
  if [[ ${INPUT_MODE} == "image" ]] || [[ ${INPUT_MODE} == "image_rna" ]]
    then
      N_SAMPLES=13                                                       # xxx image files for THYM; xxx rna-seq samples (xxx cases); xxx have both (a small number of cases have two rna-seq samples)
      N_EPOCHS=30                                                        # ignored in test mode
      BATCH_SIZE="25"                                                    # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
      PCT_TEST=".2"                                                      # proportion of samples to be held out for testing
      FINAL_TEST_BATCH_SIZE=5000                                         # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)
      TILE_SIZE="64"                                                     # must be a multiple of 64 
      TILES_PER_IMAGE="100"                                               # Training mode only. <450 for Moodus 128x128 tiles. (this parameter is automatically calculated in 'just_test mode')
      SUPERGRID_SIZE=1                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      NN_TYPE_IMG="VGG11"                                                # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
#     NN_TYPE_IMG="DCGANAE128"                                                
      RANDOM_TILES="True"                                                # select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      LEARNING_RATE=".001"
      CANCER_TYPE="THYM"
      CANCER_TYPE_LONG="Stomach_Intestine_Adenocarcinoma"      
#      CLASS_NAMES="diffuse                            stomach_NOS                 mucinous                                   intestinal_NOS                   tubular                     signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            other        mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  other       intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      CLASS_NAMES="diffuse                            mucinous                                    tubular                                   signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      STAIN_NORMALIZATION="NONE"                                         # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # ..in scattergram view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FIGURE_WIDTH=9
      FIGURE_HEIGHT=9

      NN_TYPE_RNA="DENSE"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 1)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 2)
      N_GENES=506                                                        # 60482 genes in total for THYM rna-sq data of which 506 map to PMCC gene panel genes
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/THYM_genes_of_interest
      GENE_DATA_NORM="NONE"                                              # supported options are NONE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG2PLUS1"                                    # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="False"                                             # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000"                               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="2000"                                        # only used for AEDENSE and DENSE at the moment
      GENE_EMBED_DIM="2000"                                              # only used for AEDENSE at the moment
      SHOW_ROWS=1000
      SHOW_COLS=100
  elif [[ ${INPUT_MODE} == "rna" ]] || [[ ${INPUT_MODE} == "image_rna" ]]   
    then                                                                  # Also works well  HIDDEN_LAYER_NEURONS="700"; NN_DENSE_DROPOUT_1="0.2" <<< TRY IT AGAIN
                                                                          # Also works well  HIDDEN_LAYER_NEURONS="250"; NN_DENSE_DROPOUT_1="0.2"  << BEST SO FAR?
      N_SAMPLES="479"                                                       # 479 rna-seq samples (474 cases); 229 have both (a small number of cases have two rna-seq samples)
      N_EPOCHS=300
      BATCH_SIZE="19 19 19 19 19"                                         #  number of samples in each "mini batch"
#      BATCH_SIZE="95 95 95 95 95 95 95 95 95"
      PCT_TEST="0.2"                                                      # proportion of samples to be held out for testing
#      LEARNING_RATE=".0008"
      LEARNING_RATE=".0002"                                               # learning rate for back propagation
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_transcripts_of_interest  # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/THYM_genes_of_interest        # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
                                                                         # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="True"                                              # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
#      NN_TYPE_RNA="AELINEAR"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDEEPDENSE"                                             # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="TTVAE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDENSE"                                                 # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
       NN_TYPE_RNA="DENSE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      HIDDEN_LAYER_ENCODER_TOPOLOGY="7000 6000 6000 6000"               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000 8000"                          # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="1100"                                        # only used for AEDENSE and DENSE at the moment
      GENE_EMBED_DIM="2000"                                              # only used for AEDENSE at the moment
      NN_DENSE_DROPOUT_1="0.2"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 1)
#     NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
#      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      CANCER_TYPE="THYM"
      CANCER_TYPE_LONG="Stomach_Intestine_Adenocarcinoma"      
#      CLASS_NAMES="diffuse                            stomach_NOS                 mucinous                                   intestinal_NOS                   tubular                     signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  adenocarcinoma_NOS  intestinal_adenocarcinoma_-_mucinous_type  intestinal_adenocarcinoma_-_NOS  intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
#      CLASS_NAMES="diffuse                            other        mucinous                                    tubular                                   signet_ring"
#      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  other       intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      CLASS_NAMES="diffuse                            mucinous                                    tubular                                   signet_ring"
      LONG_CLASS_NAMES="adenocarcimoa_-_diffuse_type  intestinal_adenocarcinoma_-_mucinous_type    intestinal_adenocarcinoma_-_tubular_type  stomach_adenocarcinoma_-_signet_ring"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=40
      FIGURE_HEIGHT=60

      NN_TYPE_IMG="VGG11"                                                    # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
      TILE_SIZE="128"                                                    # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=100                                                # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      SUPERGRID_SIZE=1                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                          # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                  # Show scattergram          view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                            #   In scattergram          view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                 # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=5000                                         # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

            
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
  fi  
fi

if [[ ${DATASET} == "tcl" ]]
  then
  if [[ ${INPUT_MODE} == "image" ]] || [[ ${INPUT_MODE} == "image_rna" ]]
    then
      N_SAMPLES=9
      N_EPOCHS=100
      BATCH_SIZE="9"                                                     # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
      PCT_TEST=.2                                                        # proportion of samples to be held out for testing
      TILE_SIZE="64"                                                     # must be a multiple of 64 
      TILES_PER_IMAGE=256                                                # Training mode only. <450 for Moodus 128x128 tiles. (this parameter is automatically calculated in 'just_test mode')
      SUPERGRID_SIZE=6                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      FINAL_TEST_BATCH_SIZE=250                                         # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)
      NN_TYPE_IMG="VGG11"                                                # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
#     NN_TYPE_IMG="DCGANAE128"                                                
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      LEARNING_RATE=".001"
      CANCER_TYPE="TCL"
      CANCER_TYPE_LONG="T-Cell Lymphoma"      
      CLASS_NAMES="not_tumour tumour"
      LONG_CLASS_NAMES="not_tumour tumour"
      STAIN_NORMALIZATION="NONE"                                         # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                            # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                         # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                 # Show scattergram view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                           # ..in scattergram view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                              # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FIGURE_WIDTH=9
      FIGURE_HEIGHT=9

      NN_TYPE_RNA="DENSE"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
      NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 1)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in DENSE() (parameter 2)
      N_GENES=506                                                        # 60482 genes in total for STAD rna-sq data of which 506 map to PMCC gene panel genes
      REMOVE_UNEXPRESSED_GENES="True"                                    # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                 # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest
      TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest
      GENE_DATA_NORM="NONE"                                              # supported options are NONE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG2PLUS1"                                    # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="False"                                             # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000"                               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="2000"                                        # only used for AEDENSE and DENSE at the moment
      GENE_EMBED_DIM="2000"                                              # only used for AEDENSE at the moment
      SHOW_ROWS=1000
      SHOW_COLS=100
  elif [[ ${INPUT_MODE} == "rna" ]] || [[ ${INPUT_MODE} == "image_rna" ]]   # Works well --->  N_SAMPLES=479; BATCH_SIZE="95"; PCT_TEST=.2; LEARNING_RATE=".0008"; HIDDEN_LAYER_NEURONS="1100"; NN_DENSE_DROPOUT_1="0.2;  GENE_DATA_NORM="JUST_SCALE" 
    then
      N_SAMPLES="479"                                                       # 479 rna-seq samples (474 cases); 229 have both (a small number of cases have two rna-seq samples)
      N_EPOCHS=100
      BATCH_SIZE="95"               # In 'test mode', BATCH_SIZE and SUPERGRID_SIZE determine the size of the patch, via the formula SUPERGRID_SIZE^2 * BATCH_SIZE
      PCT_TEST=.2                                                          # proportion of samples to be held out for testing
      LEARNING_RATE=".0008"
#     LEARNING_RATE=".1 .08 .03 .01 .008 .003 .001 .0008"
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_transcripts_of_interest  # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      #TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest        # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
      REMOVE_UNEXPRESSED_GENES="True"                                     # create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
      REMOVE_LOW_EXPRESSION_GENES="True"                                  # create and then apply a filter to remove genes whose value is less than or equal to LOW_EXPRESSION_THRESHOLD value *for every sample*
      LOW_EXPRESSION_THRESHOLD=1
      A_D_USE_CUPY='True'                                                # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
      COV_THRESHOLD=1.5                                                  # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
      CUTOFF_PERCENTILE=1                                                # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
                                                                         # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
      COV_UQ_THRESHOLD=0                                                 # minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
      DO_COVARIANCE="False"                                              # Should covariance  calculation be performed ? (analyse_data mode)
      DO_CORRELATION="True"                                              # Should correlation calculation be performed ? (analyse_data mode)    
      GENE_DATA_NORM="JUST_SCALE"                                        # supported are NONE JUST_SCALE GAUSSIAN
      GENE_DATA_TRANSFORM="LOG10PLUS1"                                   # supported are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1. LOG10PLUS1 is often a good choice where variance spans orders of magnitude
#      NN_TYPE_RNA="AELINEAR"                                                # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDEEPDENSE"                                             # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="TTVAE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      NN_TYPE_RNA="AEDENSE"                                                 # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
       NN_TYPE_RNA="DENSE"                                                   # supported options are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5, DENSE, DENSEPOSITIVE, AEDENSE, AEDENSEPOSITIVE, AEDEEPDENSE, TTVAE, DCGAN128
#      HIDDEN_LAYER_ENCODER_TOPOLOGY="7000 6000 6000 6000"               # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
      HIDDEN_LAYER_ENCODER_TOPOLOGY="8000 8000"                          # structure of hidden layers for AEDEEPDENSE and TTVAE only. The last value is taken as the required number of latent variables (rather than any other config variable)
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                       # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      ENCODER_ACTIVATION="none"                                          # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="1000"                                        # only used for AEDENSE and DENSE at the moment
      GENE_EMBED_DIM="2000"                                              # only used for AEDENSE at the moment
#     NN_DENSE_DROPOUT_1="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 1)
#     NN_DENSE_DROPOUT_1="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_1="0.10 0.15 0.20 0.25 0.3"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_DENSE_DROPOUT_2="0.0"                                           # percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
      NN_OPTIMIZER="ADAM"                                                # supported options are ADAM, ADAMAX, ADAGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
      CANCER_TYPE="TCL"
      CANCER_TYPE_LONG="T-Cell Lymphoma"      
      CLASS_NAMES="not_tumour tumour"
      LONG_CLASS_NAMES="not_tumour tumour"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=40
      FIGURE_HEIGHT=60
      NN_TYPE_IMG="VGG11"                                                    # for NN_MODE="gtexv6" supported are VGG11, VGG13, VGG16, VGG19, INCEPT3, LENET5; for NN_MODE="gtexv6" supported are DCGANAE128
      TILE_SIZE="128"                                                    # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=100                                                # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      SUPERGRID_SIZE=1                                                   # test mode: defines dimensions of 'super-patch' that combinine multiple batches into a grid for display in Tensorboard
      RANDOM_TILES="True"                                                # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORMALIZATION="NONE"                                          # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                             # Show annotated tiles      view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                          # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                  # Show scattergram          view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                            #   In scattergram          view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                 # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                               # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
            
  else
      echo "VARIABLES.SH: INFO: no such mode ''"
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

                                                       
USE_TILER='internal'                                                     # PGD 200318 - internal=use the version of tiler that's integrated into trainlent5; external=the standalone bash initiated version
#TILE_SIZE=299                                                           # PGD 202019 - Inception v3 requires 299x299 inputs (or does it? Other sizes seem to work - are the images being padded or trucnated by pytorch?)

LABEL_SWAP_PERUNIT="0.0"
MAKE_GREY_PERUNIT=0.0                                                    # make this proportion of tiles greyscale. used in 'dataset.py'. Not related to MINIMUM_PERMITTED_GREYSCALE_RANGE

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

CLASS_NUMPY_FILENAME="class.npy"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
