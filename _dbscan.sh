#! /bin/bash

#  See 'conf/variables' for other paramaters.  The following only descibes the shell level user options. Many additional configuration parameters are invoked in the conf file

# -d the dataset to use, specified as a TCGA cancer type code (here, "Stomach/Adenocarcinoma" which has the code "stad")
# -i means "image" mode; indicates that the input will be 3 x W x W tiles. The program itself will generate all tiles from slide image tiles as a first step.
# -a specifies the neural network to use. Here we use AE3LAYERCONV2D (and autoencoder) to generate the image embeddings.  Clustering does not use a neural network, so the -a flag is omitted
# -s if True means SKIP tiling. We use this on the clustering step because the tiles were generated at the preceeding step (line 21)
# -g if True means SKIP generation (of the pytorch dataset)
# -h highest class number to use. See conf/variables CLASS_NAMES. Here we use it to omit the 'normal' subtype (because there are fresh frozen rather than formalin fixed images) as well as the two 'NOS' categories.
# -l clustering algorithm to be used. Here, we apply the 'dbscan' clustering algorithm to the autoencoder embeddings generated in the first to runs  (line 15 and line 21)
# -u if True means use the Autoencoder output rather than the default, which is to use the pytorch datase
# -n = the network mode to use. Here, 'pre_compress initially to generate the embeddings using an Autoencoder, and then the default 'dlbcl_image' mode to run dbscan clustering on the Autoencoder output
# -c designate the subset of the cases to use, by specifying an appropriate flag. At the has to be NOT_A_MULTIMODE_CASE_FLAG for pre_compress mode (for not good reason) 
# -v means segment the cases (case files) to generate the case flags (like NOT_A_MULTIMODE_CASE_FLAG).  Only need to do this one time.


./do_all.sh     -d stad  -i image -h 4 -s False -g False -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG  

echo -en "\007"



./just_test.sh  -d stad  -i image -h 4 -s False -g False -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG

echo -en "\007"; sleep 0.2; echo -en "\007"



./do_all.sh     -d stad  -i image -h 4 -s True  -g False -n dlbcl_image                      -u True  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan

echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"
