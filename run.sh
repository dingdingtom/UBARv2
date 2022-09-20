#!/bin/bash
cd ./train/
bash run_train.sh

cd ../evaluate/
bash run_evaluate.sh

cd ./metric
bash run_new_metric.sh