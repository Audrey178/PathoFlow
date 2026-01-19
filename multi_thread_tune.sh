#!/usr/bin/bash

# --- CONFIGURATION ---
MAX_JOBS=3  # <--- Set your number of threads here

learning_rates=(1e-5)
weight_decays=(1e-4)
dropouts=(0.2)
n_masked_patches=(5)      # Reduced for example
mask_drops=(0.5)              # Reduced for example

count=0
total=$(( ${#learning_rates[@]} * ${#weight_decays[@]} * ${#dropouts[@]} * ${#n_masked_patches[@]} * ${#mask_drops[@]} ))

echo "Total combinations: $total"
echo "Running up to $MAX_JOBS jobs in parallel..."

for lr in "${learning_rates[@]}"; do
  for wd in "${weight_decays[@]}"; do
    for drop in "${dropouts[@]}"; do
      for n_mask in "${n_masked_patches[@]}"; do
        for m_drop in "${mask_drops[@]}"; do
      
          ((count++))
          run_name="vit_lr${lr}_wd${wd}_drop${drop}_nmask${n_mask}_mdrop${m_drop}"

          # --- CONCURRENCY CONTROL ---
          # Check number of background jobs. If >= MAX_JOBS, wait.
          while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            sleep 5
          done

          echo "Starting run $count/$total: $run_name"
          SAVE_DIR="./huy_tune/${run_name}"
          # Run Python in background (&)
          python main_huy.py --config-name baseline \
            lr=$lr \
            weight_decay=$wd \
            dropout=$drop \
            n_masked_patch=$n_mask \
            mask_drop=$m_drop \
            save_name=$run_name \
            results_dir_path=$SAVE_DIR \
            > "$SAVE_DIR/train.log" 2>&1
            
        done
      done
    done
  done
done

# Wait for all remaining background jobs to finish before exiting
wait
echo "All jobs finished."