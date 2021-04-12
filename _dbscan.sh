#! /bin/bash

./do_all.sh     -d stad  -i image -s False -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG  -v True            # has to be NOT_A_MULTIMODE_CASE_FLAG for pre_compress mode;   s = False = don't skip tiling;  v = True = Segement the cases (to crease the NOT_A_MULTIMODE_CASE_FLAG cases etc.)  
./just_test.sh  -d stad  -i image -s False -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG                     # has to be NOT_A_MULTIMODE_CASE_FLAG for pre_compress mode    s = False = don't skip tiling
./do_all.sh     -d stad  -i image -s True  -n dlbcl_image    -a VGG11          -u True  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan          # u = use autoencoder output (that we created in 'just_test')  s = True  = skip tiling
