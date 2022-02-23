#!/bin/bash

# exit if any command fails
# set -e

echo ""
echo ""

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

# Defaults. These can be changed via the Bash run-string - e.g. "./blahblah.sh  -d stad  -i image  -S 30 -f 5  -T 64  -b 32  -B 100  -q 0.5  -w 1.0  -h 7  -x 5  -o 2  -O 1  -a AEVGG16  -3  0.05  -t 50  -l cuda_tsne"
AE_ADD_NOISE="False"
BATCH_SIZE="47"
BATCH_SIZE_TEST="36"
CASES="ALL_ELIGIBLE_CASES"                                                                                 # default value. Possibly changed by user '-c' option. DON'T CHANGE THIS DEFAULT. OTHER VALUES GENERATE AND LEAVE FLAGS IN PLACE WHICH CAN CAUSE CONFUSION IF FORGOTTEN ABOUT!
CLUSTERING="NONE"                                                                                          # default value. Possibly changed by user '-l' option. Supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
DATASET="stad"
DIVIDE_CASES="False"                                                                                       # default value. Possibly changed by user '-v' option
ENCODER_ACTIVATION="none"                                                                                  # (no getopts option) activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
EPSILON="0.5"                                                                                         
GENE_DATA_NORM="NONE"
GENE_DATA_TRANSFORM="LOG10PLUS1"                                                                           # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1 RANKED
GENE_EMBED_DIM="100"
HIDDEN_LAYER_NEURONS="1100"
HIGHEST_CLASS_NUMBER="7"
INPUT_MODE="image"
JUST_CLUSTER="False"
JUST_TEST="False"
LEARNING_RATE=".0007"
MAKE_GREY_PERUNIT="0.0"                                                                                    # (no getopts option) Proportion of tiles to convert to greyscale
METRIC="manhattan"                                                                                         
MIN_CLUSTER_SIZE="10"
MULTIMODE="NONE"                                                                                           # default value. Possibly changed by user '-7' option
NN_DENSE_DROPOUT_1="0.2"                                                                                   # default value. Possibly changed by user '-n' option
NN_DENSE_DROPOUT_2="0.0"                                                                                   # (no getopts option) percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
NN_MODE="dlbcl_image"                                                                                      # default value. Possibly changed by user '-n' option
NN_OPTIMIZER="ADAM"                                                                                        # supported options are ADAM, ADAMAX, ADAGRAD, ADAMW, ADAMW_AMSGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
NN_TYPE_IMG="VGG11"                                                                                        # default value. Possibly changed by user '-a' option
NN_TYPE_RNA="DENSE"                                                                                        # default value. Possibly changed by user '-z' option
N_CLUSTERS="5"                                                                                             # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
N_EPOCHS="150"                                                                                             # default value. Possibly changed by user '-o' option
N_EPOCHS_TEST="1"
N_ITERATIONS="250"                                                                                         # default value. Possibly changed by user '-t' option
N_TESTS="1"                                                                                                # default value. Possibly changed by user '-Z' option # (test mode only) Number of examples to put through the model when just_test=='True'
N_SAMPLES=999
PCT_TEST=".2"
PCT_TEST___JUST_TEST="1.0"
PCT_TEST___TRAIN="0.1"
PEER_NOISE_PERUNIT="0.0"
PERPLEXITY="30."
PRETRAIN="False"        
SKIP_GENERATION="False"                                                                                    
RENDER_CLUSTERING="False"
REPEAT=1                                                                                    
SKIP_RNA_PREPROCESSING="False"
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_TRAINING="False"
SUPERGRID_SIZE="4"
TILES_PER_IMAGE="10"
TILE_SIZE="32"
USE_AUTOENCODER_OUTPUT="False"
                                                                                                           # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
HIDDEN_LAYER_ENCODER_TOPOLOGY="900 200"
STAIN_NORMALIZATION='NONE'
                                                                                                           # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
USE_UNFILTERED_DATA="False"                                                                                   # Don't filter genes (use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists)
TARGET_GENES_REFERENCE_FILE="just_hg38_protein_coding_genes"                                               # file specifying genes to be used if USE_UNFILTERED_DATA=False 
TARGET_GENES_REFERENCE_FILE_NAME="just_hg38_protein_coding_genes"                                          # To allow "data_comp.sh" to pass in just the file name, so that the user does not need to specify the whole path

REMOVE_LOW_EXPRESSION_GENES="True"                                                                         # DELETE AT CONVENIENCE
LOW_EXPRESSION_THRESHOLD=0.5                                                                               # DELETE AT CONVENIENCE

RANDOM_GENES_COUNT=0

COV_THRESHOLD=0                                                                                         # (standard deviations) only genes with at least CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
CUTOFF_PERCENTILE=0                                                                                       # Lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values

DO_COVARIANCE="False"                                                                                       # used by "analyse_data". Should covariance  calculation be performed ? (analyse_data mode only)
DO_CORRELATION="False"                                                                                      # used by "analyse_data". Should correlation calculation be performed ? (analyse_data mode only)    
A_D_USE_CUPY="True"                                                                                        # used by "analyse_data". if True, use cupy linear algrebra library rather than numpy. Only works if computer has a CUDA compatible GPU    
REMOVE_UNEXPRESSED_GENES="True"                                                                            # used by "analyse_data". create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
COV_UQ_THRESHOLD=2                                                                                         # used by "analyse_data". minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
SHOW_ROWS=1000                                                                                             # used by "analyse_data". 
SHOW_COLS=100                                                                                              # used by "analyse_data". 

while getopts a:A:b:B:c:C:d:D:e:E:f:F:g:G:h:H:i:I:j:J:k:K:l:L:m:M:n:N:o:O:p:P:q:Q:r:R:s:S:t:T:u:U:v:V:w:W:x:X:y:Y:z:Z:0:1:2:3:4:5:6:7:8:9: option
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};;                                                                             
    A) AE_ADD_NOISE=${OPTARG};;                                                                             
    b) BATCH_SIZE=${OPTARG};;                                                                             
    B) BATCH_SIZE_TEST=${OPTARG};;                                                                             
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    C) MIN_CLUSTER_SIZE=${OPTARG};;
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    D) TARGET_GENES_REFERENCE_FILE_NAME=${OPTARG};;
    e) EPSILON=${OPTARG};;                                                                                 # supported: any of the sklearn metrics
    E) GENE_EMBED_DIM=${OPTARG};;                                                                          # supported: in most cases, one of the sklearn metrics (but not cuda_tsne, which only supports Euclidean)
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    F) HIDDEN_LAYER_ENCODER_TOPOLOGY=${OPTARG};;                                                           # structure of hidden layers (DEEPDENSE, AEDEEPDENSE and TTVAE only. The number of neurons for the final layer is taken from GENE_EMBED_DIMS
    g) SKIP_GENERATION=${OPTARG};;                                                                         # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    G) SUPERGRID_SIZE=${OPTARG};;                                                                          
    h) HIGHEST_CLASS_NUMBER=${OPTARG};;                                                                    # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    H) HIDDEN_LAYER_NEURONS=${OPTARG};;                                                                    
    i) INPUT_MODE=${OPTARG};;                                                                              
    I) USE_UNFILTERED_DATA=${OPTARG};;
    j) JUST_TEST=${OPTARG};;                                                                               
    J) JUST_CLUSTER=${OPTARG};;                                                                             
    k) REMOVE_UNEXPRESSED_GENES=${OPTARG};;
    K) COV_UQ_THRESHOLD=${OPTARG};;
    l) CLUSTERING=${OPTARG};;                                                                              # supported: NONE, otsne, sk_tsne, cuda_tsne, sk_agglom, sk_spectral, hdbscan, dbscan
    L) LEARNING_RATE=${OPTARG};;                                                                           
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    M) METRIC=${OPTARG};;                                                                                  # supported: any of the sklearn metrics. Only 'euclidean' in the case of cuda_tsne
    n) NN_MODE=${OPTARG};;                                                                                 # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    N) SKIP_TRAINING=${OPTARG};;                                                                           # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    o) N_EPOCHS=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    O) N_EPOCHS_TEST=${OPTARG};;                                                                           # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    p) PERPLEXITY=${OPTARG};;                                                                              
    P) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    q) PCT_TEST___TRAIN=${OPTARG};;                                                                        
    Q) SHOW_COLS=${OPTARG};;
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    R) REPEAT=${OPTARG};;                                                                                  # number of times to repeat the experiment
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True,   skip tiling (to save - potentially quite a lot of time - if the desired tiles already exists)
    S) N_SAMPLES=${OPTARG};;                                                                             
    t) N_ITERATIONS=${OPTARG};;                                                                            # Number of iterations. Used by clustering algorithms only (neural networks use N_EPOCHS)
    T) TILE_SIZE=${OPTARG};;
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. if 'True', use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    U) SHOW_COLS=${OPTARG};; 
    v) DIVIDE_CASES=${OPTARG};;                                                                             
    V) DO_COVARIANCE=${OPTARG};;
    w) PCT_TEST___JUST_TEST=${OPTARG};;                                                                    
    W) DO_CORRELATION=${OPTARG};;
    x) N_CLUSTERS=${OPTARG};;                                                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    X) SKIP_RNA_PREPROCESSING=${OPTARG};;                                                                  
    y) A_D_USE_CUPY=${OPTARG};;
    Y) TARGET_GENES_REFERENCE_FILE=${OPTARG};;    
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    Z) N_TESTS=${OPTARG};;                                                                             
    0) STAIN_NORMALIZATION=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    2) RANDOM_GENES_COUNT=${OPTARG};;                                                                             
    3) PEER_NOISE_PERUNIT=${OPTARG};;                                                                      
    4) NN_OPTIMIZER=${OPTARG};; 
    5) GENE_DATA_TRANSFORM=${OPTARG};; 
    6) GENE_DATA_NORM=${OPTARG};; 
    7) NN_DENSE_DROPOUT_1=${OPTARG};; 
    8) COV_THRESHOLD=${OPTARG};; 
    9) CUTOFF_PERCENTILE=${OPTARG};; 
    esac
  done
  
source conf/variables.sh

TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/${TARGET_GENES_REFERENCE_FILE_NAME}


if [[ ${PRETRAIN} == "True" ]]; 
  then
    SKIP_TILING=True
    SKIP_GENERATION=True
fi

echo "===> STARTING"

echo "=====> STEP 1 OF 3: CLEANING DATASET"

# maybe clear case subsetting flags
if [[ ${DIVIDE_CASES} == 'True' ]]; then
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'HAS_IMAGE'"
  find ${DATA_DIR} -type f -name HAS_IMAGE                                    -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'HAS_BOTH'"
  find ${DATA_DIR} -type f -name HAS_BOTH                                     -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE____MATCHED'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE____MATCHED                      -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'MULTIMODE____TEST'"
  find ${DATA_DIR} -type f -name MULTIMODE____TEST                            -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE                                 -delete                                             # it's critical that existing  NON-MULTIMODE cases flags are deleted, otherwise the image mode run and the rna mode run won't choose the same cases
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE____IMAGE'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE____IMAGE                       -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE____IMAGE_TEST'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE____IMAGE_TEST                  -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE____RNA'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE____RNA                         -delete
  #~ echo "DO_ALL.SH: INFO: recursively deleting flag files              matching this pattern:  'UNIMODE_CASE____RNA_TEST'"
  find ${DATA_DIR} -type f -name UNIMODE_CASE____RNA_TEST                    -delete
fi

if [[ ${SKIP_TILING} == "False" ]]; 
  then
    if [[ ${REGEN} == "regen" ]]; 
      then
        echo "=====> REGENERATING DATASET FROM SOURCE DATA - THIS CAN TAKE A LONG TIME (E.G. 20 MINUTES)"
        rm -rf ${DATA_DIR}
        rsync -ah --info=progress2 ${DATASET}/ ${DATA_DIR}
      else
        #~ echo "=====> DELETING All PRE-PROCEESSING FILES AND LEAVING JUST SVS AND UQ FILES"
        #~ echo "DO_ALL.SH: INFO: deleting all empty subdirectories under '${DATA_DIR}'"
        find ${DATA_DIR} -type d -empty -delete
        #~ echo "DO_ALL.SH: INFO: deleting the 'SUFFICIENT_SLIDES_TILED' flag"        
        rm "${DATA_DIR}/SUFFICIENT_SLIDES_TILED" > /dev/null 2>&1
        #~ echo "DO_ALL.SH: INFO: deleting all 'SLIDE_TILED' flags"        
        find ${DATA_DIR} -type f -name "SLIDE_TILED"          -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting subdirectories matching this pattern:  '${FLAG_DIR_SUFFIX}'"
        find ${DATA_DIR} -type d -name ${FLAG_DIR_SUFFIX}          -exec rmdir {} \;  
        #~ echo "DO_ALL.SH: INFO: recursively deleting residual                  '.tar' files"
        find ${DATA_DIR} -type f -name "*.tar"                     -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting residual                  '.gz'  files"
        find ${DATA_DIR} -type f -name "*.gz"                      -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting                           '.fqln'            files created in earlier runs"
        find ${DATA_DIR} -type l -name "*.spcn"                    -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting                           '.spcn'            files created in earlier runs"
        find ${DATA_DIR} -type l -name "*.fqln"                    -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting                           'entire_patch.npy' files created in earlier runs"
        find ${DATA_DIR} -type f -name "entire_patch.npy"          -delete 
        #~ echo "DO_ALL.SH: INFO: recursively deleting files                      matching this pattern:  '${RNA_NUMPY_FILENAME}'"
        find ${DATA_DIR} -type f -name ${RNA_NUMPY_FILENAME}       -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting files                      matching this pattern:  '*${RNA_FILE_REDUCED_SUFFIX}'"
        #~ find ${DATA_DIR} -type f -name *${RNA_FILE_REDUCED_SUFFIX} -delete
        #~ echo "DO_ALL.SH: INFO: recursively deleting files                      matching this pattern:  '${CLASS_NUMPY_FILENAME}'"
        find ${DATA_DIR} -type f -name ${CLASS_NUMPY_FILENAME}     -delete
        
        
        if [[ ${INPUT_MODE} == 'image' ]]; then
            #~ echo "DO_ALL.SH: INFO: image       mode, so recursively deleting existing image     embedding files ('${EMBEDDING_FILE_SUFFIX_IMAGE}')"
            find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE}      -delete
        elif [[ ${INPUT_MODE} == 'rna' ]]; then
            #~ echo "DO_ALL.SH: INFO: rna         mode, so recursively deleting existing rna       embedding files ('${EMBEDDING_FILE_SUFFIX_RNA}')"
            find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_RNA}        -delete
        elif [[ ${INPUT_MODE} == "image_rna" ]]; then
            #~ echo "DO_ALL.SH: INFO: 'image_rna' mode, so recursively deleting existing image_rna embedding files ('${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}')"
            find ${DATA_DIR} -type f -name *${EMBEDDING_FILE_SUFFIX_IMAGE_RNA}  -delete
        fi
        
        if [[ ${INPUT_MODE} == "image" ]]; then
            #~ echo "DO_ALL.SH: INFO: 'image' mode, so deleting saved image indices:  train_inds_image, test_inds_image"
            rm ${DATA_DIR}/train_inds_image  > /dev/null 2>&1
            rm ${DATA_DIR}/test_inds_image   > /dev/null 2>&1
            echo "DO_ALL.SH: INFO: recursively deleting files (tiles)           matching this pattern:  '*.png'                            <<< for image mode, deleting all the .png files (i.e. tiles) can take quite some time as there can be up to millions of tiles"
            find ${DATA_DIR} -type f -name *.png                                            -delete
        fi
        
        if [[ ${INPUT_MODE} == "rna" ]]; then
            #~ echo "DO_ALL.SH: INFO: 'image' mode, so deleting saved image indices:  train_inds_image, test_inds_image"
            rm ${DATA_DIR}/train_inds_rna    > /dev/null 2>&1
            rm ${DATA_DIR}/test_inds_rna     > /dev/null 2>&1
    fi
fi
    
    #tree ${DATA_DIR}
    cd ${BASE_DIR}

    
echo "=====> STEP 2 OF 3: PRE-PROCESS TRUTH VALUES (TRUE SUBTYPES) AND IF APPLICABLE, ALSO (i) REMOVE ROWS (RNA EXPRESSION DATA) FROM FPKM-UQ FILES THAT DO NOT CORRESPOND TO TARGET GENE LIST (ii) EXTRACT RNA EXPRESSION INFORMATION AND SAVE AS NUMPY FILES"

    if [[ ${INPUT_MODE} == "rna" ]] || [[ ${INPUT_MODE} == "image_rna" ]] ;
      then
        if [[ ${SKIP_RNA_PREPROCESSING} != "True" ]] 
          then     
            sleep ${SLEEP_TIME}
            cp ${DATASET}_global/*MASTER.csv  ${DATA_DIR}
            cp ${DATASET}_global/*of_interest ${DATA_DIR}
            cp ${DATASET}_global/just_hg38_protein_coding_genes ${DATA_DIR}
            cp ${DATASET}_global/ENSG_UCSC_biomart_ENS_id_to_gene_name_table ${DATA_DIR}      
            python reduce_FPKM_UQ_files.py --data_dir ${DATA_DIR} --target_genes_reference_file ${TARGET_GENES_REFERENCE_FILE} --rna_file_suffix ${RNA_FILE_SUFFIX} --rna_file_reduced_suffix ${RNA_FILE_REDUCED_SUFFIX}  \
            --rna_exp_column ${RNA_EXP_COLUMN} --use_unfiltered_data ${USE_UNFILTERED_DATA} --skip_generation ${SKIP_GENERATION} --random_genes_count ${RANDOM_GENES_COUNT}
  
  
            #~ echo "=====> EXTRACTING RNA EXPRESSION INFORMATION AND SAVING AS NUMPY FILES"
            sleep ${SLEEP_TIME}
            python process_rna_exp.py --data_dir ${DATA_DIR} --rna_file_suffix ${RNA_FILE_SUFFIX} --rna_file_reduced_suffix ${RNA_FILE_REDUCED_SUFFIX} --rna_exp_column ${RNA_EXP_COLUMN} \
            --rna_numpy_filename ${RNA_NUMPY_FILENAME} --use_unfiltered_data ${USE_UNFILTERED_DATA}
        fi
    fi
    
    #~ echo "=====> STEP 2B OF 3: (IF APPLICABLE) PRE-PROCESSING CLASS (GROUND TRUTH) INFORMATION AND SAVING AS NUMPY FILES"
    sleep ${SLEEP_TIME}
    #~ cp ${GLOBAL_DATA}/${DATASET}_mapping_file_MASTER ${MAPPING_FILE_NAME}     ${DATA_DIR}
    cp ${GLOBAL_DATA}/${MAPPING_FILE_NAME}                                    ${DATA_DIR}
    cp ${GLOBAL_DATA}/${ENSG_REFERENCE_FILE_NAME}                             ${DATA_DIR}  
    python process_classes.py  --data_dir ${DATA_DIR} --dataset ${DATASET} --global_data ${GLOBAL_DATA} --class_numpy_filename ${CLASS_NUMPY_FILENAME} --mapping_file ${MAPPING_FILE} --mapping_file_name ${MAPPING_FILE_NAME} --case_column ${CASE_COLUMN} --class_column=${CLASS_COLUMN}  
    
fi


echo "=====> STEP 3 OF 3: RUNNING THE NETWORK (PYTORCH DATASET WILL BE GENERATED AND TILING WILL BE PERFORMED IF IMAGE MODE, UNLESS EITHER SUPPRESSED BY USER OPTION)"
sleep ${SLEEP_TIME}
cd ${NN_APPLICATION_PATH}
CUDA_LAUNCH_BLOCKING=1 python ${NN_MAIN_APPLICATION_NAME} \
--input_mode ${INPUT_MODE} --multimode ${MULTIMODE} --just_profile ${JUST_PROFILE} --just_test ${JUST_TEST} --skip_tiling ${SKIP_TILING} --skip_generation ${SKIP_GENERATION} \
--dataset ${DATASET} --cases ${CASES} --data_dir ${DATA_DIR} --data_source ${DATA_SOURCE} --divide_cases ${DIVIDE_CASES} --cases_reserved_for_image_rna ${CASES_RESERVED_FOR_IMAGE_RNA} \
--global_data ${GLOBAL_DATA} --mapping_file_name ${MAPPING_FILE_NAME} \
--log_dir ${LOG_DIR} --save_model_name ${SAVE_MODEL_NAME} --save_model_every ${SAVE_MODEL_EVERY} \
--ddp ${DDP} --use_autoencoder_output ${USE_AUTOENCODER_OUTPUT} --ae_add_noise ${AE_ADD_NOISE} --pretrain ${PRETRAIN} \
--clustering ${CLUSTERING} --n_clusters ${N_CLUSTERS} --metric ${METRIC} --epsilon ${EPSILON} --repeat ${REPEAT} --min_cluster_size ${MIN_CLUSTER_SIZE} --perplexity ${PERPLEXITY} --momentum ${MOMENTUM} \
--rna_file_name ${RNA_NUMPY_FILENAME} --rna_file_suffix ${RNA_FILE_SUFFIX}  --use_unfiltered_data ${USE_UNFILTERED_DATA} --remove_low_expression_genes  ${REMOVE_LOW_EXPRESSION_GENES} \
--embedding_file_suffix_rna ${EMBEDDING_FILE_SUFFIX_RNA} --embedding_file_suffix_image ${EMBEDDING_FILE_SUFFIX_IMAGE} --embedding_file_suffix_image_rna ${EMBEDDING_FILE_SUFFIX_IMAGE_RNA} \
--low_expression_threshold ${LOW_EXPRESSION_THRESHOLD} --remove_unexpressed_genes ${REMOVE_UNEXPRESSED_GENES} --target_genes_reference_file ${TARGET_GENES_REFERENCE_FILE} \
--do_covariance ${DO_COVARIANCE} --do_correlation ${DO_CORRELATION} --a_d_use_cupy ${A_D_USE_CUPY} --cov_threshold ${COV_THRESHOLD} --cov_uq_threshold ${COV_UQ_THRESHOLD} --cutoff_percentile ${CUTOFF_PERCENTILE} \
--class_numpy_file_name ${CLASS_NUMPY_FILENAME} --highest_class_number ${HIGHEST_CLASS_NUMBER} \
--nn_mode ${NN_MODE} --use_same_seed ${USE_SAME_SEED} --nn_type_img ${NN_TYPE_IMG} --nn_type_rna ${NN_TYPE_RNA}  \
--nn_dense_dropout_1 ${NN_DENSE_DROPOUT_1} --nn_dense_dropout_2 ${NN_DENSE_DROPOUT_2} \
--encoder_activation ${ENCODER_ACTIVATION} --optimizer ${NN_OPTIMIZER} --n_samples ${N_SAMPLES} --pct_test ${PCT_TEST} --n_tests ${N_TESTS} --final_test_batch_size ${FINAL_TEST_BATCH_SIZE} \
--gene_data_norm ${GENE_DATA_NORM} --gene_data_transform ${GENE_DATA_TRANSFORM} --gene_embed_dim ${GENE_EMBED_DIM} --hidden_layer_neurons ${HIDDEN_LAYER_NEURONS} --hidden_layer_encoder_topology ${HIDDEN_LAYER_ENCODER_TOPOLOGY} \
--cancer_type ${CANCER_TYPE} --cancer_type_long ${CANCER_TYPE_LONG} --class_names ${CLASS_NAMES} --long_class_names ${LONG_CLASS_NAMES} --class_colours ${CLASS_COLOURS} --colour_map ${COLOUR_MAP} \
--n_tiles ${TILES_PER_IMAGE} --rand_tiles ${RANDOM_TILES} --tile_size ${TILE_SIZE} --zoom_out_mags ${ZOOM_OUT_MAGS} --zoom_out_prob ${ZOOM_OUT_PROB} \
--n_epochs ${N_EPOCHS} --n_iterations ${N_ITERATIONS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
--latent_dim ${LATENT_DIM} --max_consecutive_losses ${MAX_CONSECUTIVE_LOSSES} --min_uniques ${MINIMUM_PERMITTED_UNIQUE_VALUES} \
--greyness ${MINIMUM_PERMITTED_GREYSCALE_RANGE} --make_grey_perunit ${MAKE_GREY_PERUNIT}  --peer_noise_perunit ${PEER_NOISE_PERUNIT} --label_swap_perunit ${LABEL_SWAP_PERUNIT} \
--target_tile_offset ${TARGET_TILE_OFFSET} --stain_norm ${STAIN_NORMALIZATION} --stain_norm_target ${STAIN_NORM_TARGET} --min_tile_sd ${MIN_TILE_SD}  --points_to_sample ${POINTS_TO_SAMPLE} \
--show_rows ${SHOW_ROWS} --show_cols ${SHOW_COLS} --figure_width ${FIGURE_WIDTH} --figure_height ${FIGURE_HEIGHT} --annotated_tiles ${ANNOTATED_TILES} --supergrid_size ${SUPERGRID_SIZE} \
--patch_points_to_sample ${PATCH_POINTS_TO_SAMPLE} --scattergram ${SCATTERGRAM} --box_plot ${BOX_PLOT} --minimum_job_size ${MINIMUM_JOB_SIZE} --show_patch_images ${SHOW_PATCH_IMAGES} \
--bar_chart_x_labels=${BAR_CHART_X_LABELS} --bar_chart_sort_hi_lo=${BAR_CHART_SORT_HI_LO}  --bar_chart_show_all=${BAR_CHART_SHOW_ALL} \
--probs_matrix ${PROBS_MATRIX} --probs_matrix_interpolation ${PROBS_MATRIX_INTERPOLATION} 
cd ${BASE_DIR}
