#!/bin/bash

source conf/variables.sh

CPUS = $(grep ^cpu\\scores /proc/cpuinfo | uniq)

echo "  START.SH: INFO:      setting up " ${CPUS} " threads"
for  i in `seq 1 ${CPUS}`;
  do
    bash save_svs_to_tiles.sh ${i} ${CPUS} &
  done
 

wait

NUMBER_OF_TILES=$(find ${EXPERIMENT_DIR} -name *${TILE_SIZE}.png | wc -l)  # USING ${TILE_SIZE} as all files generated by save_svs_to_tiles.py have this in their filenames (and there are other irrelevant files in the PATCH_PATH directory)
echo "START.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}
