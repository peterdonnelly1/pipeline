#!/bin/bash

# exit if any command fails
# set -e

echo ""
echo ""

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

# Defaults. These can be changed via the Bash run-string - e.g. "./blahblah.sh  -d stad  -i image  -S 30 -f 5  -T 64  -b 32  -B 100  -q 0.5  -w 1.0  -h 7  -x 5  -o 2  -O 1  -a AEVGG16  -3  0.05  -t 50  -l cuda_tsne"
DATASET="stad"
INPUT_MODE="image"
BATCH_SIZE="36"
BATCH_SIZE_TEST="36"
PCT_TEST=".2"
PCT_TEST___TRAIN="0.1"
PCT_TEST___JUST_TEST="1.0"
MULTIMODE="NONE"                                                                                           # possibly changed by user '-m' argument if required, but it needs an initial value
TILES_PER_IMAGE="100"
TILE_SIZE="64"
N_EPOCHS="100"                                                                                             # possibly changed by user '-o' argument if required, but it needs an initial value
N_ITERATIONS="250"                                                                                         # possibly changed by user '-t' argument if required, but it needs an initial value
NN_MODE="dlbcl_image"                                                                                      # possibly changed by user '-n' argument if required, but it needs an initial value
NN_TYPE_IMG="AEVGG16"  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< # possibly changed by user '-a' argument if required, but it needs an initial value
NN_TYPE_RNA="AEDENSE"  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< # possibly changed by user '-z' argument if required, but it needs an initial value
CASES="UNIMODE_CASE"                                                                                 # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                                                       # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"        
CLUSTERING="NONE"                                                                                          # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
N_CLUSTERS="5"                                                                                             # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
METRIC="manhattan"                                                                                         
EPSILON="0.5"                                                                                         
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="True"  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PEER_NOISE_PCT="0.0"
MAKE_GREY_PCT="0.0"
N_SAMPLES="310"
MIN_CLUSTER_SIZE="10"
PERPLEXITY="31 32"
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
GENE_DATA_TRANSFORM="LOG10PLUS1" 
GENE_DATA_NORM="NONE"
COV_THRESHOLD="0.0"                                                                                        # (standard deviations) Only genes with >CUTOFF_PERCENTILE % across samples having rna-exp values above COV_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
CUTOFF_PERCENTILE="0"                                                                                      # lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher COV_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values
                                                                                                           # It's better to filter with the combination of CUTOFF_PERCENTILE/COV_THRESHOLD than wth COV_UQ_THRESHOLD because the former is computationally much faster
HIDDEN_LAYER_ENCODER_TOPOLOGY="200"
STAIN_NORMALIZATION='NONE'


while getopts a:A:b:B:c:C:d:e:E:f:F:g:G:h:H:i:j:J:l:L:m:M:n:N:o:O:p:P:q:r:R:s:S:t:T:u:v:w:x:X:z:0:1:3:4:5:6:7:8:9: option
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
    e) EPSILON=${OPTARG};;                                                                                 # supported: any of the sklearn metrics
    E) GENE_EMBED_DIM=${OPTARG};;                                                                          # supported: in most cases, one of the sklearn metrics (but not cuda_tsne, which only supports Euclidean)
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    F) HIDDEN_LAYER_ENCODER_TOPOLOGY=${OPTARG};;                                                           # structure of hidden layers (DEEPDENSE, AEDEEPDENSE and TTVAE only. The number of neurons for the final layer is taken from GENE_EMBED_DIMS
    g) SKIP_GENERATION=${OPTARG};;                                                                         # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    G) SUPERGRID_SIZE=${OPTARG};;                                                                          
    h) HIGHEST_CLASS_NUMBER=${OPTARG};;                                                                    # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    H) HIDDEN_LAYER_NEURONS=${OPTARG};;                                                                    
    i) INPUT_MODE=${OPTARG};;                                                                              # supported: image, rna, image_rna
    j) JUST_TEST=${OPTARG};;                                                                               
    J) JUST_CLUSTER=${OPTARG};;                                                                             
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
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    R) RENDER_CLUSTERING=${OPTARG};;                                                                       # 'True'   or 'False'. if 'True', show plots on terminal (they are always be saved to logs)
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True,   skip tiling (to save - potentially quite a lot of time - if the desired tiles already exists)
    S) N_SAMPLES=${OPTARG};;                                                                             
    t) N_ITERATIONS=${OPTARG};;                                                                            # Number of iterations. Used by clustering algorithms only (neural networks use N_EPOCHS)
    T) TILE_SIZE=${OPTARG};;
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. if 'True', use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    v) DIVIDE_CASES=${OPTARG};;                                                                             
    w) PCT_TEST___JUST_TEST=${OPTARG};;                                                                    
    x) N_CLUSTERS=${OPTARG};;                                                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    X) SKIP_RNA_PREPROCESSING=${OPTARG};;                                                                  
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    0) STAIN_NORMALIZATION=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    3) PEER_NOISE_PCT=${OPTARG};;                                                                      
    4) MAKE_GREY_PCT=${OPTARG};; 
    5) GENE_DATA_TRANSFORM=${OPTARG};; 
    6) GENE_DATA_NORM=${OPTARG};; 
    7) NN_DENSE_DROPOUT_1=${OPTARG};; 
    8) COV_THRESHOLD=${OPTARG};; 
    9) CUTOFF_PERCENTILE=${OPTARG};; 
    esac
  done

i=1

for GENE_EMBED_DIM_VALUE in "2000"

do

  echo "CUDA_TSNE_MULTI_RUN.SH:  run number           = "${i}     "                                   <<< dataset generation and tiling (if image) are performed on the first run only"
  echo "CUDA_TSNE_MULTI_RUN.SH:  GENE_EMBED_DIM_VALUE = "${GENE_EMBED_DIM_VALUE} "for this run"
      
  if [[ ${i} -eq 1 ]]
  
    then 
      
      rm logs/lowest_loss_ae_model.pt
      
      # training (first time)
     ./do_all.sh  -d ${DATASET}  -i ${INPUT_MODE}  -S ${N_SAMPLES}  -o ${N_EPOCHS}  -f ${TILES_PER_IMAGE}  -T ${TILE_SIZE}   -b ${BATCH_SIZE}       -1 ${PCT_TEST___TRAIN}      -h ${HIGHEST_CLASS_NUMBER}   -s False   \
-X ${SKIP_RNA_PREPROCESSING}  -g False   -j False  -n pre_compress  -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -E ${GENE_EMBED_DIM_VALUE}  -v ${DIVIDE_CASES}  -A ${AE_ADD_NOISE}  -c ${CASES}                                         \
-3 ${PEER_NOISE_PCT} -4 ${MAKE_GREY_PCT} -u False
 
    else

      # training (subsequent times)
      echo "CUDA_TSNE_MULTI_RUN.SH:  run = "${i} " tiling (iff image) and generation will be skipped"
     rm logs/lowest_loss_ae_model.pt
    ./do_all.sh  -d ${DATASET}  -i ${INPUT_MODE}   -S ${N_SAMPLES}  -o ${N_EPOCHS} -f ${TILES_PER_IMAGE}  -T ${TILE_SIZE}   -b ${BATCH_SIZE}       -1 ${PCT_TEST___TRAIN}      -h ${HIGHEST_CLASS_NUMBER}    -s True    \
-X ${SKIP_RNA_PREPROCESSING}  -g True   -j False  -n pre_compress   -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -E ${GENE_EMBED_DIM_VALUE}   -A ${AE_ADD_NOISE}  -c ${CASES}                                      \
-3 ${PEER_NOISE_PCT} -4 ${MAKE_GREY_PCT} -u False 

  fi
    
  rm logs/ae_output_features.pt
  
  echo ""
  echo ""
  echo "CUDA_TSNE_MULTI_RUN.SH:  generate embeddings using best model produced and saved during training (test mode is invoked by ' -j True' user option)"
  
  ./do_all.sh  -d ${DATASET}  -i ${INPUT_MODE}   -S ${N_SAMPLES}  -o ${N_EPOCHS_TEST} -f ${TILES_PER_IMAGE}  -T ${TILE_SIZE}   -b ${BATCH_SIZE_TEST}  -1 ${PCT_TEST___JUST_TEST}  -h ${HIGHEST_CLASS_NUMBER}  -s True   \
-X True  -g True  -j True   -n pre_compress     -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -E ${GENE_EMBED_DIM_VALUE} -A False -u True  -c ${CASES}
  

  # cluster and display
  ./do_all.sh  -d ${DATASET}  -i ${INPUT_MODE}   -a ${NN_TYPE_IMG}  -E ${GENE_EMBED_DIM_VALUE} -t 5000  -s True  -g True  -n dlbcl_image  -c ${CASES}  -l cuda_tsne  -p "1 2 3 7 10 20 30 70 100"  \
-G ${SUPERGRID_SIZE}  -R ${RENDER_CLUSTERING} -P ${PRETRAIN}

  i=$((i+1))  
  
done


echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"


