#!/bin/bash

source conf/variables.sh

echo "===> STARTING"
echo "=====> STEP 1 OF 7: DELETING EXTRANEOUS FILES FROM DATA TREE"

tree dataset_mini


sleep ${SLEEP_TIME}
cd ${BASE_DIR}

echo "=====> STEP 2 OF 7: GENERATING TILES FROM SVS IMAGES"
sleep ${SLEEP_TIME}
cd ${BASE_DIR}
./start.sh


echo "===> FINISHED "
cd ${BASE_DIR}
