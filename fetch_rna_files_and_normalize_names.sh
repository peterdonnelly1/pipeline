#! /bin/bash

# rename each gene expression results file to a name which incorporates the corresponding SVS filename. This is now we match the gene expression files (which unfortunately don't use the TCGA naming convention) to the SVS files.
# MAPPING_FILE file contains the (manually constructed) mapping between gene expression files and svs files for a given case. In the (tedious) manual matching process, TCGA case ID is used to match the two files. 

source conf/variables.sh

bold=$(tput bold)
normal=$(tput sgr0)

echo "FETCH_AND_NORMALIZE: INFO: "${bold}"!!! remember to save the mapping file spreadsheet as a CSV file !!!!! " ${normal}
echo "==========> STEP 2A: FETCH_AND_NORMALIZE: INFO: about to copy rna-seq  files from the biodata directory for this experiment )"${bold}${RNA_PATH}${normal}") to intermediate directory 1 for this experiment ("${bold}${INTERMEDIATE_DIR_1}${normal}
sleep .5

# copy normalized gene expression results files from the applicable biodata directory (~/biodata/<experiment name>/rna) to our working directory (~/rna/gene_results_files/)

rm -rf ${INTERMEDIATE_DIR_1}/*
find ${RNA_PATH} -name "*genes.results" -exec cp -vf {} ${INTERMEDIATE_DIR_1} \;
NUMBER_COPIED=$(find ${INTERMEDIATE_DIR_1} -name "*.results" | wc -l)
echo "FETCH_AND_NORMALIZE: INFO: finished copying rna-seq files to working directory "${bold}${INTERMEDIATE_DIR_1}${normal}
echo "FETCH_AND_NORMALIZE: INFO: Number of rna_seq files                copied to "${bold}${INTERMEDIATE_DIR_1}${normal}"                  = "${bold}${NUMBER_COPIED}${normal}


# rename rna-expression files by giving them the SVS filename as a prefix (source of renaming information is MAPPING_FILE)
echo "==========> STEP 2B: FETCH_AND_NORMALIZE: INFO: about to prefix  rna-seq files with SVS filename (source of SVS file names is "${bold}${MAPPING_FILE}${normal}")"
sleep .5
cd ${INTERMEDIATE_DIR_1}
rm -rf ${INTERMEDIATE_DIR_2}/*
awk -F',' 'system("cp -fv " $3 " " "'"$INTERMEDIATE_DIR_2"'" "/" $2 "___" $3)' ${MAPPING_FILE}
NUMBER_RNA_SEQ_NORMALIZED=$(find ${INTERMEDIATE_DIR_2} -name "*.results" | wc -l)
echo "FETCH_AND_NORMALIZE: INFO: finished copying rna-seq files to working directory "${bold}${INTERMEDIATE_DIR_2}${normal}
echo "FETCH_AND_NORMALIZE: INFO: Number of rna_seq files renamed and copied to    "${bold}${INTERMEDIATE_DIR_2}${normal}" = "${bold}${NUMBER_RNA_SEQ_NORMALIZED}${normal}"  (Note: this number will be larger than the preceeding if some cases have more than one SVS image)"


# create files containing cancer type (type_n) and save to a file with very similar normalized name used for rna-seq files above (source of renaming information is MAPPING_FILE)
echo "==========> STEP 2C: FETCH_AND_NORMALIZE: INFO: about to generate files containing type_n (numberic representation of the cancer type)"
sleep .5
awk -v INT_DIR=$INTERMEDIATE_DIR_2 -F',' 'system("echo " $4 " > " INT_DIR "/" $2 "___" $3 ".type_n.csv" )' ${MAPPING_FILE}
NUMBER_TYPE_N_CREATED=$(find ${INTERMEDIATE_DIR_2} -name "*.results" | wc -l)
echo "FETCH_AND_NORMALIZE: INFO: finished creating type_n.csv files in  "${bold}${INTERMEDIATE_DIR_2}${normal}
echo "FETCH_AND_NORMALIZE: INFO: Number of type_n.csv files created in "${bold}${INTERMEDIATE_DIR_2}${normal}" = "${bold}${NUMBER_TYPE_N_CREATED}${normal}
