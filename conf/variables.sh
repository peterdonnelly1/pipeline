#!/bin/bash

alias cls='printf "\033c"'

EXPERIMENT=gtexv6													    # use gtexv6 for dlbcl in dpcca mode and the 'dlbcl' dataset
#EXPERIMENT=eye													        # use eye for    dlbcl in dpcca mode and the 'eye'   dataset
#EXPERIMENT=dlbcl_image											        # use dlbcl  for dlbcl in image mode and the 'dlbcl' dataset
DATA_ROOT=data

NN_APPLICATION_PATH=dpcca
#NN_MAIN_APPLICATION_NAME=traindpcca.py                                  # use traindpcca.py for dlbcl in dpcca mode
#NN_DATASET_HELPER_APPLICATION_NAME=data.gtexv6.generate                 # use gtexv6        for dlbcl in dpcca mode
NN_MAIN_APPLICATION_NAME=trainlenet5.py                                # use traindpcca.py for dlbcl in image (lenet5) mode
NN_DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate           # use dlbcl_image   for dlbcl in image (lenet5) mode
N_EPOCHS=99999
BATCH_SIZE=256
LATENT_DIM=1
MAX_CONSECUTIVE_LOSSES=20
#LEARNING_RATE=.001

TILES_TO_GENERATE_PER_SVS=100                                           # set up so that ALL tiles will be used by the dlbcl python "generate.py" function
TILE_SIZE=128                                                           # PGD 200108 - correct for gtexv6 experiment. It does not work with any old tile size, so be careful
INCLUDE_WHITE_TILES=0                                                   # ignore 'white' tiles
WHITENING_THRESHOLD=0.05                                                # definition of a white tile. 0 means 100% of tiles must not be white; 0.05 means 95% of tiles must not be white etc

# main paths
BASE_DIR=/home/peter/git/pipeline
EXPERIMENT_DIR=${BASE_DIR}/${DATA_ROOT}
SVS_PATH=${EXPERIMENT_DIR}
RNA_PATH=${EXPERIMENT_DIR}
PATCH_PATH=${EXPERIMENT_DIR}

# variabes used by shells scripts which process gene files
MAPPING_FILE=${EXPERIMENT_DIR}/mapping_file.csv
INTERMEDIATE_DIR_1=${EXPERIMENT_DIR}/gene_results_files
INTERMEDIATE_DIR_2=${EXPERIMENT_DIR}/gene_results_files_normalized_names
GENE_EXPRESSION_CSV_FILES_PATH=${EXPERIMENT_DIR}/tcga_dlbc_rna_seq_results
RNA_FILE_PATTERN="*FPKM-UQ.txt"
RNA_NUMPY_FILENAME="FPKM-UQ.npy"
TISSUE_NUMPY_FILENAME=tissue_class.npy

# The list of case_ids you want to download heatmaps from
#CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt
#DATA_PATH=${BASE_OUT}/training_data                                    # Change this to your training data folder
#DATA_LIST='tumor_data_list_toy.txt'                                    # Text file to contain subfolders for testing (1st line), training (the rest)
#MODEL='RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_cancer_tils_none_1117_1811_0.9157633018398808_9.t7'
