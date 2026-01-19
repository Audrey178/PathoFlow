#!/usr/bin/bash

echo "*********************************"
echo "Startting experiments with baseline"
echo "*********************************"

feats_path="datasets/feats_resnet"
python eval.py --config-name baseline hidden_size=2048 feats_path=${feats_path} csv_path="datasets/csv"

echo "Finished"
echo ""s