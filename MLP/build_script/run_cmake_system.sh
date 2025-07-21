#!/bin/bash

start=$(date +%s)

cd ../
mkdir ./build
cd ./build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j16

end=$(date +%s)

echo Build tme was  $(($end-$start)) seconds.

