#!/bin/bash

if [ ! -f matrix-gen ]; then
    echo "matrix-gen is not compiled. Please compile it first."
    exit
fi

SIZES=(2 4 8 16 32 64 128 256 512 1024)

printf "\033[1mMatrix Generator\033[0m\n"
echo "----------------------------------------"
echo "Starting matrix generation"

for x in ${SIZES[@]}; do
    printf "    \033[35mGenerating matrices of size \
\033[1m$x\033[22m x \033[1m$x\033[0m\n"
    ./matrix-gen $x
done

echo
printf "\033[32mDone!\033[0m\n"

