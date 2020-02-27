#!/bin/bash

source conf/contrast_play_variables.sh

echo "=====> STEP 1 OF 1: GENERATING TILES FROM SVS IMAGES"
sleep ${SLEEP_TIME}
cd ${BASE_DIR}
./contrast_play_start.sh

echo "===> FINISHED "
cd ${BASE_DIR}
