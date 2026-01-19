#!/bin/bash

TOP_K_LIST=(4 8 16 32 64 128 256 512 1024 2048 4096 8192)

echo "*********************************"
echo "Starting experiments with baseline"
echo "*********************************"

for TOP_K in "${TOP_K_LIST[@]}"; do
    echo "---------------------------------"
    echo "Running experiment with top_k=${TOP_K}"
    echo "---------------------------------"

    run_name="baseline_topk${TOP_K}"

    python main.py \
        --config-name baseline \
        top_k=${TOP_K} \
        save_name=$run_name

    echo "Finished top_k=${TOP_K}"
    echo ""
done

echo "*********************************"
echo "All experiments finished"
echo "*********************************"
