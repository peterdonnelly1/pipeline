#!/bin/bash

rm -rf dataset
rm -rf 0008
mkdir -v 0008


find stad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find coad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find kidn -name '*UQ*' -exec cp --parents \{\} 0008 \;
find sarc -name '*UQ*' -exec cp --parents \{\} 0008 \;
find luad -name '*UQ*' -exec cp --parents \{\} 0008 \;
#find igg_ -name '*UQ*' -exec cp --parents \{\} 0008 \;

mv  0008/stad/* 0008
mv  0008/coad/* 0008
mv  0008/kidn/* 0008
mv  0008/sarc/* 0008
mv  0008/luad/* 0008
#mv  0008/igg_/* 0008

rmdir -v 0008/stad
rmdir -v 0008/coad
rmdir -v 0008/kidn
rmdir -v 0008/sarc
rmdir -v 0008/luad
#rmdir -v 0008/igg_

./create_master.sh 0008
