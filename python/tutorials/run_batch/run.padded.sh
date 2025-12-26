#!/bin/bash

# Run to get log with different configurations
# Usage: bash run.padded.log.sh

echo "Start runs"

export TRITON_HIP_USE_ASYNC_COPY=1

## this is for collecting log
#export MLIR_ENABLE_DUMP=1 TRITON_DISABLE_LINE_INFO=1 AMDGCN_ENABLE_DUMP=1

#for BM in 128; do
#    for BN in 128; do
#        for BK in 64; do
#            for nonKDim in 16 32; do
for BM in 32 64 128; do
    for BN in 32 64 128; do
        for BK in 32 64 128; do
            for nonKDim in 16 32; do
                echo "Running: BM=${BM}, BN=${BN}, BK=${BK}, nonKDim=${nonKDim}"
                #python test.matmul.padded.bash.py --BM ${BM} --BN ${BN} --BK ${BK} --nonKDim ${nonKDim} &> log.bm${BM}.bn${BN}.bk${BK}.nonkdim${nonKDim}.log
                #echo "Completed: log.bm${BM}.bn${BN}.bk${BK}.nonkdim${nonKDim}.log"
                # this is for collecting rocprof
                rocprofv3 -i config.json -d att_bm${BM}_bn${BN}_bk${BK}_nonkdim${nonKDim} -- python test.matmul.padded.bash.py --BM ${BM} --BN ${BN} --BK ${BK} --nonKDim ${nonKDim}
            done
        done
    done
done

echo "All runs completed!"
