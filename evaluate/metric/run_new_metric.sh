#!/bin/bash
# e2e modeling

input_path='../../train/experiments_all2/all_sd-0_lr-0.00015_bs-1_am-8/model/epoch8_domain-all'
input_file_name='inference-test-standard-gen-state.json' 
output_father_folder_path='result/best_test'
output_file_name='OUTPUT.json'
python evaluate.py --bleu --success --input_path=$input_path --input_file_name=$input_file_name --output_father_folder_path=$output_father_folder_path --output_file_name=$output_file_name && echo $input_path is done