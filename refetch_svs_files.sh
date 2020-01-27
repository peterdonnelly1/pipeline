#! /bin/bash

SVS_FILES_SOURCE_FOLDER="/home/peter/biodata/svs"
SVS_DATA_FOLDER="/home/peter/git/quip_classification_data/svs"

find ${SVS_FILES_SOURCE_FOLDER} -name "*.svs" -exec cp -vf {} ${SVS_DATA_FOLDER} \;
