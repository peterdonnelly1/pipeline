#!/bin/bash

source conf/variables.sh

bold=$(tput bold)
normal=$(tput sgr0)

echo "COPY_RNA_FILES:  INFO: intermediate directory containing rna-seq files with normalized names  = " ${bold}${INTERMEDIATE_DIR_2}${normal}
echo "COPY_RNA_FILES:  INFO: directory to which rna-seq files will be copied                        = " ${bold}${PATCH_PATH}${normal}
echo ""

LINE_N=0

for file in ${INTERMEDIATE_DIR_2}/*.*; do

    LINE_N=$((LINE_N+1))

    echo "COPY_RNA_FILES:  INFO: now copying file: "${bold}${LINE_N}${normal}   
    echo "COPY_RNA_FILES:  INFO: fully qualified rna filename: " ${file}    
    filename="${file##*/}"
    echo "COPY_RNA_FILES:  INFO: just rna filename:            " ${filename}
        
    # The file name of the rna-seq file was constructed so as to include the complete name of the SVS file at the start. The SVS filename is also the name of the sub-directory of 'patches' that we want to copy the rna-sef file to.
    TARGET_PATCHES_SUBDIR=`echo ${file} | sed -e 's!^.*TCGA\(.*\)___.*!TCGA\1!'`
    
    cp -vf ${file} ${PATCH_PATH}/${TARGET_PATCHES_SUBDIR}
    
    if [ $? -ne 0 ]; then
      touch ${PATCH_PATH}/${TARGET_PATCHES_SUBDIR}/rna_files_copied_ok.txt
    fi

    echo ""

done

NUMBER_COPIED=$(find ${PATCH_PATH}     -name "*.results" | wc -l)
echo "COPY_RNA_FILES:  INFO: Number of rna expression info files saved to "${bold}${PATCH_PATH}${normal}"                  = "${bold}${NUMBER_COPIED}${normal}

exit 0;

