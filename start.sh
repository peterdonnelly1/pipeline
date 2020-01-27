#!/bin/bash

source conf/variables.sh


# Launch 31 processes, one for each hard thread on Ryzen 2950X
bash save_svs_to_tiles.sh 0 1

wait

NUMBER_OF_TILES=$(find ${PATCH_PATH} -name *${TILE_SIZE}.png | wc -l)  # USING ${TILE_SIZE} as all files generated by save_svs_to_tiles.py have this in their filenames (and there are other irrelevant files in the PATCH_PATH directory)
echo "START.SH: INFO: total number of tiles = " ${NUMBER_OF_TILES}
