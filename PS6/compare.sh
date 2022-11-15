#!/bin/bash

MAX_ALLOWABLE_PERCENT="0.05"
MAX_EPSILONS=16

help_message() {
    echo "Help Message: "
    echo "./compare.sh [version]"
    echo "[version]: cuda or cublas"
}

if [[ $# -le 0 ]]; then
    help_message
    exit
fi

VERSION=$1

# Enables / Disables Color Output
USE_SGR=1

if [ $USE_SGR ]; then
    CYAN_TEXT=$'\033[36m'
    GREEN_TEXT=$'\033[32m'
    RED_TEXT=$'\033[31m'
    RESET=$'\033[0m'


    CROSS=✗
    TICK=✔
fi

if [ ! -f compare/compare ]; then
    echo "Comparison program is not compiled. Please compile it."
    exit
fi

if [ ! -f parallel_${VERSION} ]; then
    echo "Parallel implementation not compiled. Please compile it."
    exit
fi

SIZES=(2 4 8 16 32 64 128 256 512 1024)

for x in ${SIZES[@]}; do

    MAX_ALLOWABLE=$(echo "$x * $x" | bc -l)
    MAX_ALLOWABLE=$(echo "$MAX_ALLOWABLE * $MAX_ALLOWABLE_PERCENT" | bc)
    MAX_ALLOWABLE=${MAX_ALLOWABLE%.*}

    if [ -z $MAX_ALLOWABLE ]; then
        MAX_ALLOWABLE=0
    fi

    echo ""
    echo "==================== SIZE $x ===================="
    if [ ! -f data/size${x}-C.bin ]; then
        echo "> Computing serial result matrix"
        ./serial $x > /dev/null
    fi
    echo "> Running the parallel implementation for size $x"
    printf "${CYAN_TEXT}Program Output:${RESET}\n"



    exec 3>&1
    exec 1> >(sed -E "s/(.+)/\t${CYAN_TEXT}\1${RESET}/g")

    ./parallel_${VERSION} $x

    exec 1>&3 3>&-

    REF_FILE=data/size${x}-C.bin
    SOLN_FILE="data/size${x}-C-${VERSION}.bin"

    ERRORS=$(./compare/compare ${REF_FILE} ${SOLN_FILE} ${x} ${MAX_EPSILONS})

    if [[ $ERRORS -gt $MAX_ALLOWABLE ]]; then
        printf "${RED_TEXT}${CROSS} FAILED WITH $ERRORS ERRORS${RESET}\n"
    else
        printf "${GREEN_TEXT}${TICK} Solution valid with $ERRORS errors${RESET}\n"
    fi

done
