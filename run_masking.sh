#!/bin/bash

# ==========================================
# HYPERPARAMETERS FOR MASKING
# ==========================================

# 1. n_masked_patch: Number of patches/tokens to mask (Integer)
# Adjust these values based on your total sequence length (top_k)
N_MASKED_LIST=(4 8 12)

# 2. mask_drop: Dropout probability for the mask (Float)
MASK_DROP_LIST=(0.5)

# Fixed Config Name (change 'baseline' if your config file is named differently)
CONFIG_NAME="baseline"

echo "*********************************"
echo "Starting Masking Grid Search"
echo "*********************************"

# Loop through Number of Masked Patches
for N_MASKED in "${N_MASKED_LIST[@]}"; do
    
    # Loop through Mask Dropout Rates
    for MASK_DROP in "${MASK_DROP_LIST[@]}"; do
        
        echo "---------------------------------"
        echo "Experiment: n_masked_patch=${N_MASKED} | mask_drop=${MASK_DROP}"
        echo "---------------------------------"

        # Create a unique save name for WandB/Logging
        # This helps you filter runs easily in the WandB dashboard
        run_name="mask_n${N_MASKED}_drop${MASK_DROP}"

        python main.py \
            --config-name ${CONFIG_NAME} \
            n_masked_patch=${N_MASKED} \
            mask_drop=${MASK_DROP} \
            save_name=${run_name}

        echo "Finished run: ${run_name}"
        echo ""
    done
done

echo "*********************************"
echo "All masking experiments finished"
echo "*********************************"