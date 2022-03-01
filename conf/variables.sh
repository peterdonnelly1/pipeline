#!/bin/bash
#set -e

alias cls='printf "\033c"'
SLEEP_TIME=0

# main directory paths & file names

NN_APPLICATION_PATH=dpcca
BASE_DIR=/home/peter/git/pipeline                                                                          # root directory for everything (shell scripts, code, datasets, logs ...)
DATA_ROOT=dataset                                                                                          # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}                                                                          # location of the above. Not to be confused with DATA_SOURCE, which points to the master directory (via ${DATASET})
DATA_SOURCE=${BASE_DIR}/${DATASET}                                                                         # structured directory containing dataset. A copy is made to DATA_ROOT. DATA_SOURCE is left untouched
GLOBAL_DATA=${BASE_DIR}/${DATASET}_global                                                                  # name of a custom mapping file, if one exists, else "none"
MAPPING_FILE_NAME=${DATASET}_mapping_file_MASTER.csv                                                       # mapping file to use, if it's a special one. (Default "mapping_file" (no extension), doesn't have to be specified)
MAPPING_FILE=${DATA_DIR}/${MAPPING_FILE_NAME}
LOG_DIR=${BASE_DIR}/logs

# 'pre-sets'

if [[ ${NN_MODE} == "dlbcl_image" ]]
  then
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_dlbcl_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py 
    MAIN_APPLICATION_NAM=trainlenet5.py                                                                    # use trainlenet5.py   for any "images + classes" dataset
    DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image                                        # use generate_images  for any "images + classes" dataset OTHER THAN MNIST
elif [[ ${NN_MODE} == "pre_compress" ]]
    then
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_pre_compress_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
    MAIN_APPLICATION_NAM=pre_compress.py                                                                   # use pre_compress.py   for pre-compressing a dataset
    DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate                                             # use pre_compress      for pre-compressing a dataset
elif [[ ${NN_MODE} == "analyse_data" ]]
    then
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_analyse_data_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
    A_D_USE_CUPY='True'                                                                                    # whether or not to use cupy (instead of numpy). cupy is roughly the equivalent of numpy, but supports NVIDIA GPUs
    MAIN_APPLICATION_NAM=analyse_data.py                               
    DATASET_HELPER_APPLICATION_NAME=data.pre_compress.generate           
elif [[ ${NN_MODE} == "gtexv6" ]]
    then  
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_gtexv6_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
elif [[ ${NN_MODE} == "mnist" ]]
    then  
    SKIP_GENERATION="True"
    cp -f ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py_mnist_version  ${BASE_DIR}/${NN_APPLICATION_PATH}/data/__init__.py
    MAIN_APPLICATION_NAM=traindpcca.py                                                                     # use traindpcca.py    for the  MNIST "digit images + synthetic classes" dataset  
#    DATASET_HELPER_APPLICATION_NAME=data.mnist.generate                                                   # use generate_mnist   for the  MNIST "digit images + synthetic classes" dataset
    DATASET_HELPER_APPLICATION_NAME=data.dlbcl_image.generate_image                                        # use generate_images  for any "images + classes" dataset OTHER THAN MNIST    
else
    echo "VARIABLES.SH: INFO: no such INPUT_MODE as '${INPUT_MODE}' for dataset ${DATASET}"
fi


if [[ ${DATASET} == "stad" ]]; 
  then

    CANCER_TYPE="STAD"
    CANCER_TYPE_LONG="Stomach_Adenocarcinoma"      

  if [[ ${INPUT_MODE} == "image" ]]
    then
      HIGHEST_CLASS_NUMBER=8                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1
      FINAL_TEST_BATCH_SIZE=2                                                                              # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
      RANDOM_TILES="True"                                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
      CLASS_NAMES="diffuse   tubular   mucinous    signet_ring    papillary   tubular  stomach_NOS    intestinal_NOS       none"
      LONG_CLASS_NAMES="diffuse   tubular   mucinous    signet_ring    papillary   tubular  stomach_NOS    intestinal_NOS       none"
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"     # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"

      # Vizualization related
      ANNOTATED_TILES="False"                                                                              # Show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
      SCATTERGRAM="True"                                                                                   # Show scattergram image in tensorboard
      SHOW_PATCH_IMAGES="False"                                                                            # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="False"                                                                                 # Supplement scattergram with a probabilities matrix image in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # Interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      PATCH_POINTS_TO_SAMPLE=500                                                                           # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
      FIGURE_WIDTH=8
      FIGURE_HEIGHT=8

      ENCODER_ACTIVATION="none"                                                                            # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      HIDDEN_LAYER_NEURONS="2000"                                                                          # only used for AEDENSE and DENSE at the moment
      SHOW_ROWS=1000
      SHOW_COLS=100

  # caution: Not all stad cases have both image and rna-seq samples. Further, for rna-seq, there are only five subtypes with meaningful numbers of rna-seq cases (hence HIGHEST_CLASS_NUMBER=4)

  elif [[ ${INPUT_MODE} == "rna" ]]  
    then
      HIGHEST_CLASS_NUMBER=4                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1

      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/just_hg38_protein_coding_genes 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/STAD_genes_of_interest                                    # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                                                         # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      #~ GENE_EMBED_DIM="100"                                                                              # only used for AEDENSE at the moment
      CLASS_NAMES="diffuse tubular mucinous intest_nos adeno_nos"
      LONG_CLASS_NAMES="diffuse tubular mucinous intest_nos adeno_nos"
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12
      TILE_SIZE="128"                                                                                      # On Moodus, 50 samples @ 8x8 & batch size 64 = 4096x4096 is Ok
      TILES_PER_IMAGE=1234                                                                                 # Training mode only (automatically calculated as SUPERGRID_SIZE^2 * BATCH_SIZE for just_test mode)
      RANDOM_TILES="True"                                                                                  # Select tiles at random coordinates from image. Done AFTER other quality filtering
      #~ STAIN_NORMALIZATION="NONE"                                                                        # options are NONE, reinhard, spcn  (specifies the type of stain colour normalization to be performed)
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                                                  # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=141                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

  elif [[ ${INPUT_MODE} == "image_rna" ]]  
    then                                                                 
      #~ GENE_EMBED_DIM="100"                                                                              # only used for AEDENSE at the moment
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
      RANDOM_TILES="True"                                                                                  # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL STAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # Show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # In scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                                                  # Show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=100                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

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
      HIGHEST_CLASS_NUMBER=1                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1
      FINAL_TEST_BATCH_SIZE=2                                                                              # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
      RANDOM_TILES="True"                                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"

      # Vizualization related
      ANNOTATED_TILES="False"                                                                              # Show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
      SCATTERGRAM="True"                                                                                   # Show scattergram image in tensorboard
      SHOW_PATCH_IMAGES="False"                                                                            # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="False"                                                                                 # Supplement scattergram with a probabilities matrix image in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # Interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      PATCH_POINTS_TO_SAMPLE=500                                                                           # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
      FIGURE_WIDTH=8
      FIGURE_HEIGHT=8

      ENCODER_ACTIVATION="none"                                                                            # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      SHOW_ROWS=1000
      SHOW_COLS=100


  elif [[ ${INPUT_MODE} == "rna" ]]  
    then
      HIGHEST_CLASS_NUMBER=1                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/just_hg38_protein_coding_genes 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/COAD_genes_of_interest                                    # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                                                         # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      #~ GENE_EMBED_DIM="100"                                                                              # only used for AEDENSE at the moment

      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12

      RANDOM_TILES="True"                                                                                  # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # in scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      

      PROBS_MATRIX="True"                                                                                  # show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=141                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

  elif [[ ${INPUT_MODE} == "image_rna" ]]  
    then                                                                 
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12
      RANDOM_TILES="True"                                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL COAD SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # in scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                                                  # show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=100                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)


  else
      echo "VARIABLES.SH: INFO: no such input mode as ${INPUT_MODE}"
  fi  



elif [[ ${DATASET} == "thym" ]]; 
  then
  
    CANCER_TYPE="THYM"
    CANCER_TYPE_LONG="Thymoma"   
    CLASS_NAMES="type_A type_AB type_B1 type_B2 type_B3 type_C"
    LONG_CLASS_NAMES="type_A type_AB type_B1 type_B2 type_B3 type_C"


  if [[ ${INPUT_MODE} == "image" ]]
    then
      HIGHEST_CLASS_NUMBER=5                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1
      FINAL_TEST_BATCH_SIZE=2                                                                              # number of batches of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated). Don't make it too large because it's passed through as a single super-batch.
      RANDOM_TILES="True"                                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
#     STAIN_NORM_TARGET="0f344863-11cc-4fae-8386-8247dff59de4/TCGA-BR-A4J6-01Z-00-DX1.59317146-9CAF-4F48-B9F6-D026B3603652.svs"   # <--THIS IS A RANDOMLY CHOSEN SLIDE FROM THE MATCHED SUBSET 
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL THYM SET & THE TARGET_TILE_COORDS COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"

      # Vizualization related
      ANNOTATED_TILES="False"                                                                              # Show annotated tiles image in tensorboard (use SCATTERGRAM for larger numbers of tiles. ANNOTATED_TILES generates each tile as a separate subplot and can be very slow and also has a much lower upper limit on the number of tiles it can handle)
      SCATTERGRAM="True"                                                                                   # Show scattergram image in tensorboard
      SHOW_PATCH_IMAGES="False"                                                                            # ..in scattergram image, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="False"                                                                                 # Supplement scattergram with a probabilities matrix image in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # Interpolate the scattergram with a probabilities matrix. Valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      PATCH_POINTS_TO_SAMPLE=500                                                                           # How many points to sample when selecting a 'good' patch (i.e. few background tiles) from the slide
      FIGURE_WIDTH=8
      FIGURE_HEIGHT=8

      ENCODER_ACTIVATION="none"                                                                            # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      SHOW_ROWS=1000
      SHOW_COLS=100


  elif [[ ${INPUT_MODE} == "rna" ]]  
    then
      HIGHEST_CLASS_NUMBER=5                                                                               # i.e. number of subtypes. Can't be greater than the number of entries in CLASS_NAMES, recalling that classes are numbered from 0, not 1
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/just_hg38_protein_coding_genes 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/pmcc_cancer_genes_of_interest 
      #~ TARGET_GENES_REFERENCE_FILE=${DATA_DIR}/THYM_genes_of_interest                                    # use to specify a specific subset of genes. Ignored if USE_UNFILTERED_DATA="True".
#      ENCODER_ACTIVATION="none sigmoid relu tanh"                                                         # activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
      #~ GENE_EMBED_DIM="100"                                                                              # only used for AEDENSE at the moment

      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12

      RANDOM_TILES="True"                                                                                  # Select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL THYM SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # in scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      

      PROBS_MATRIX="True"                                                                                  # show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=141                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)

  elif [[ ${INPUT_MODE} == "image_rna" ]]  
    then                                                                 
      SHOW_ROWS=1000
      SHOW_COLS=100
      FIGURE_WIDTH=12
      FIGURE_HEIGHT=12
      RANDOM_TILES="True"                                                                                  # select tiles at random coordinates from image. Done AFTER other quality filtering
      STAIN_NORM_TARGET="./7e13fe2a-3d6e-487f-900d-f5891d986aa2/TCGA-CG-4301-01A-01-TS1.4d30d6f5-c4e3-4e1b-aff2-4b30d56695ea.svs"   # <--THIS SLIDE IS ONLY PRESENT IN THE FULL THYM SET & THE COORDINATES BELOW ARE FOR IT
      TARGET_TILE_COORDS="5000 5500"
      ANNOTATED_TILES="False"                                                                              # show annotated tiles view in tensorboard      
      PATCH_POINTS_TO_SAMPLE=500                                                                           # test mode only: How many points to sample when selecting a 'good' (i.e. few background tiles) patch from the slide
      SCATTERGRAM="True"                                                                                   # Show scattergram     view in tensorboard      
      SHOW_PATCH_IMAGES="True"                                                                             # in scattergram       view, show the patch image underneath the scattergram (normally you'd want this)      
      PROBS_MATRIX="True"                                                                                  # show probabilities matrix view in tensorboard
      PROBS_MATRIX_INTERPOLATION="spline16"                                                                # valid values: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
      FINAL_TEST_BATCH_SIZE=100                                                                            # number of tiles to test against optimum model after each run (rna mode doesn't need this because the entire batch can easily be accommodated)


  else
      echo "VARIABLES.SH: INFO: no such input mode as ${INPUT_MODE}"
  fi  


else
    echo "VARIABLES.SH: INFO: no such dataset as '${DATASET}'"
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
ENSG_REFERENCE_COLUMN=0
RNA_EXP_COLUMN=1                                                                                           # correct for "*FPKM-UQ.txt" files (where the Gene name is in the first column and the normalized data is in the second column)
CLASS_NUMPY_FILENAME="class.npy"
CASE_COLUMN="bcr_patient_uuid"
CLASS_COLUMN="type_n"
