#!/bin/bash

SEEDS=(1026 1024 512 1356 896)

echo "*********************************"
echo "Starting experiments with baseline"
echo "*********************************"

for SEED in "${SEEDS[@]}"; do
    echo "---------------------------------"
    echo "Running experiment with ratio=${SEED}"
    echo "---------------------------------"

    run_name="baseline_ratio0.1_seed${SEED}"

    python main.py \
        --config-name baseline \
        seed=${SEED} \
        save_name=$run_name

    echo "Finished seed=${SEED}"
    echo ""
done

echo "*********************************"
echo "All experiments finished"
echo "*********************************"
