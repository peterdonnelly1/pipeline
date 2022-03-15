#!/bin/bash
set -x

rm -rf dataset
rm -rf 0008
mkdir 0008

find stad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find coad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find kidn -name '*UQ*' -exec cp --parents \{\} 0008 \;
find sarc -name '*UQ*' -exec cp --parents \{\} 0008 \;
find luad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find lusc -name '*UQ*' -exec cp --parents \{\} 0008 \;
find lgg_ -name '*UQ*' -exec cp --parents \{\} 0008 \;
find pcpg -name '*UQ*' -exec cp --parents \{\} 0008 \;
find esca -name '*UQ*' -exec cp --parents \{\} 0008 \;

mv  0008/stad/* 0008
mv  0008/coad/* 0008
mv  0008/kidn/* 0008
mv  0008/sarc/* 0008
mv  0008/luad/* 0008
mv  0008/lusc/* 0008
mv  0008/lgg_/* 0008
mv  0008/pcpg/* 0008
mv  0008/esca/* 0008

rmdir -v 0008/stad
rmdir -v 0008/coad
rmdir -v 0008/kidn
rmdir -v 0008/sarc
rmdir -v 0008/luad
rmdir -v 0008/lusc
rmdir -v 0008/lgg_
rmdir -v 0008/pcpg
rmdir -v 0008/esca

./create_master.sh 0008
