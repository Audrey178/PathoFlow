#!/bin/bash

RATIOS=(0.05 0.07 0.1 0.25 0.5 0.75)

echo "*********************************"
echo "Starting experiments with baseline"
echo "*********************************"

for RATIO in "${RATIOS[@]}"; do
    echo "---------------------------------"
    echo "Running experiment with ratio=${RATIO}"
    echo "---------------------------------"

    run_name="baseline_ratio${RATIO}"

    python main.py \
        --config-name baseline \
        ratio=${RATIO} \
        save_name=$run_name

    echo "Finished ratio=${RATIO}"
    echo ""
done

echo "*********************************"
echo "All experiments finished"
echo "*********************************"
