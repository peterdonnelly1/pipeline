#!/bin/bash

# exit if any command fails
# set -e

echo ""
echo ""

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

# Defaults. These can be changed via the Bash run-string - e.g. "./blahblah.sh  -d stad  -i image  -S 30 -f 5  -T 64  -b 32  -B 100  -q 0.5  -w 1.0  -h 7  -x 5  -o 2  -O 1  -a AEVGG16  -3  0.05  -t 50  -l cuda_tsne"
DATASET="stad"
INPUT_MODE="rna"    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< # 'analyse_data' mode only supports RNA-Seq data, not image data
BATCH_SIZE="16"
BATCH_SIZE_TEST="16"
PCT_TEST=".2"
PCT_TEST___TRAIN="0.1"
PCT_TEST___JUST_TEST="1.0"
MULTIMODE="NONE"                                                                                           # possibly changed by user '-m' argument if required, but it needs an initial value
TILES_PER_IMAGE="10"
TILE_SIZE="32"
N_EPOCHS="2"                                                                                               # possibly changed by user '-o' argument if required, but it needs an initial value
N_ITERATIONS="250"                                                                                         # possibly changed by user '-t' argument if required, but it needs an initial value
NN_MODE="analyse_data"                                                                                     # possibly changed by user '-n' argument if required, but it needs an initial value
NN_TYPE_IMG="AEVGG16"                                                                                      # possibly changed by user '-a' argument if required, but it needs an initial value
NN_TYPE_RNA="AEDENSE"                                                                                      # possibly changed by user '-z' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                                                 # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                                                       # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"        
CLUSTERING="NONE"                                                                                          # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
N_CLUSTERS="5"                                                                                             # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
METRIC="manhattan"                                                                                         
EPSILON="0.5"                                                                                         
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="True"
PEER_NOISE_PERUNIT="0.0"
MAKE_GREY_PERUNIT="0.0"
N_SAMPLES=310
MIN_CLUSTER_SIZE="10"
PERPLEXITY="30."
AE_ADD_NOISE="False"
SKIP_TRAINING="False"
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_GENERATION="False"                                                                                    
JUST_TEST="False"
JUST_CLUSTER="False"
SKIP_RNA_PREPROCESSING="False"
GENE_EMBED_DIM="100"
N_EPOCHS_TEST="1"
SUPERGRID_SIZE="4"
RENDER_CLUSTERING="False"
LEARNING_RATE=".0001"
                                                                                                           # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
USE_UNFILTERED_DATA="True"                                                                                 # Don't filter genes (use FPKM-UQ.txt files, rather than FPKM-UQ_reduced.txt (filtered) files, even if the latter exists)
TARGET_GENES_REFERENCE_FILE="just_hg38_protein_coding_genes"                                               # file specifying genes to be used if USE_UNFILTERED_DATA="False".  
TARGET_GENES_REFERENCE_FILE_NAME="just_hg38_protein_coding_genes"                                          # To allow "data_comp.sh" to pass in just the file name, so that the user does not need to specify the whole path

REMOVE_LOW_EXPRESSION_GENES="True"                                                                         # DELETE AT CONVENIENCE
LOW_EXPRESSION_THRESHOLD=0.5                                                                               # DELETE AT CONVENIENCE

DO_COVARIANCE="False"                                                                                       # used by "analyse_data". Should covariance  calculation be performed ? (analyse_data mode only)
DO_CORRELATION="False"                                                                                      # used by "analyse_data". Should correlation calculation be performed ? (analyse_data mode only)    
A_D_USE_CUPY="True"                                                                                        # used by "analyse_data". if True, use cupy linear algrebra library rather than numpy. Only works if computer has a CUDA compatible GPU    
REMOVE_UNEXPRESSED_GENES="True"                                                                            # used by "analyse_data". create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
COV_THRESHOLD="2"                                                                                          # used by "analyse_data". (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
CUTOFF_PERCENTILE="99"                                                                                     # used by "analyse_data". lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
COV_UQ_THRESHOLD=2                                                                                         # used by "analyse_data". minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
SHOW_ROWS=1000                                                                                             # used by "analyse_data". 
SHOW_COLS=100                                                                                              # used by "analyse_data". 


while getopts a:A:b:B:c:C:d:D:e:E:f:F:g:G:h:H:i:I:j:J:k:K:l:L:m:M:n:N:o:O:p:P:q:Q:r:R:s:S:t:T:u:U:v:V:w:W:x:X:y:Y:z:0:1:3:4:5:6:7:8:9: option
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
    R) RENDER_CLUSTERING=${OPTARG};;                                                                       # 'True'   or 'False'. if 'True', show plots on terminal (they are always be saved to logs)
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
    0) STAIN_NORMALIZATION=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    3) PEER_NOISE_PERUNIT=${OPTARG};;                                                                      
    4) MAKE_GREY_PERUNIT=${OPTARG};; 
    5) GENE_DATA_TRANSFORM=${OPTARG};; 
    6) GENE_DATA_NORM=${OPTARG};; 
    7) NN_DENSE_DROPOUT_1=${OPTARG};; 
    8) COV_THRESHOLD=${OPTARG};; 
    9) CUTOFF_PERCENTILE=${OPTARG};; 
    esac
  done
  
source conf/variables.sh


if [[ ${SKIP_GENERATION} == "False" ]]
  then
  
  echo "=====> SUPER-STEP I OF II: GENERATING AND SAVING A FRESH COPY OF THE GENES DATASET IN BOTH NUMPY AND CUPY FORMATS (PICKLED TO REDUCE SIZE)"
  echo ""
  echo "DELETING ANY EXISTING PICKLE FILES FROM ${BASE_DIR}/dpcca/data/analyse_data/"
  
  rm ${BASE_DIR}/dpcca/data/analyse_data/genes.pickle
  rm ${BASE_DIR}/dpcca/data/analyse_data/genes_cupy.pickle.npy
  
  ./do_all.sh -n pre_compress -d stad  -i rna  -z AEDENSE -o 1 -I ${USE_UNFILTERED_DATA} -D ${TARGET_GENES_REFERENCE_FILE_NAME}
  ./do_all.sh -n pre_compress -d stad  -i rna  -z AEDENSE -o 1 -I True

fi

echo "=====> SUPER-STEP II OF II: PERFORMING DATA COMPREHENSION"

#~ ./do_all.sh  -d ${DATASET}  -n analyse_data -i rna -8 ${COV_THRESHOLD} -9 ${CUTOFF_PERCENTILE} -K ${COV_UQ_THRESHOLD} \
#~ -V ${DO_COVARIANCE} -W ${DO_CORRELATION} -y ${A_D_USE_CUPY} -I ${USE_UNFILTERED_DATA} -D ${TARGET_GENES_REFERENCE_FILE_NAME} -X True

./do_all.sh  -d ${DATASET}  -n analyse_data -i rna -8 ${COV_THRESHOLD} -9 ${CUTOFF_PERCENTILE} -K ${COV_UQ_THRESHOLD} \
-V ${DO_COVARIANCE} -W ${DO_CORRELATION} -y ${A_D_USE_CUPY} -I True -X True
