# UBARv2

## Abstract

This paper studies the exposure bias problem in task-oriented dialog systems, where the model's generated content over multiple turns drives the dialog context away from the ground-truth distribution at training time, introducing error propagation and damaging the robustness of the TOD system.
To bridge the gap between training and inference for multi-turn task-oriented dialogs, we propose session-level sampling which explicitly exposes the model to sampled generated content of dialog context during training. Additionally, we employ a dropout based consistency regularization with masking strategy R-Mask to further improve the robustness and performance of the model. The proposed UBARv2 achieves state-of-the-art performance at the standardized evaluation benchmark MultiWOZ and extensive experiments show the effectiveness of the proposed methods. 



## Requirements

- CUDA 11.4
- Python 3.7
- PyTorch 1.7.1
- spaCy
- transformers 4.3.3
- fuzzywuzzy 0.18.0
- lexical-diversity 0.1.1
- python-Levenshtein 0.12.2
- sacrebleu 2.0.0
- sacremoses 0.0.46

 We use the tokenization tool in SpaCy which can be installed through: 

```
python -m spacy download en_core_web_sm --user
```



## Data Preprocessing

Follow [UBAR](https://github.com/TonyNemo/UBAR-MultiWOZ) data preprocessing.

The [data](https://drive.google.com/file/d/1azN8WQKpY5cvGoSaD1xc94em6U70DetQ/view?usp=sharing) is released.



## Two-Stage Training

Our implementation supports training on CPU or a single GPU or multi-GPUs.

### Stage 1

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=16

python -m torch.distributed.launch --nproc_per_node=4 model_train.py \
   --use_low_resource=0 \
   --mode=train \
   --accumulation=8 \
   --batch_size_train=1 \
   --lr=0.00005 \
   --type_scheduler=linw \
   --max_epoch=75 \
   --rate_kl=1e-2
```

### Stage 2

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=16

python -m torch.distributed.launch --nproc_per_node=4 model_train2.py \
   --use_low_resource=0 \
   --dir_pack="./experiments_all/all_sd-0_lr-5e-05_bs-1_am-8/model/epoch75_domain-all" \
   --mode=train2 \
   --accumulation=8 \
   --batch_size_train=1 \
   --lr=0.00015 \
   --type_scheduler=cosw \
   --max_epoch=10 \
   --rate_kl=1e-2 \
   --dependency=only_b \
   --p_gen_lower=0.01 \
   --p_gen_upper=0.01 \
   --str_name_element_mask_input1=bspn \
   --str_name_element_mask_input2=bspn \
   --use_gen_mask=0 \
   --use_same_mask=0 \
   --use_raw_context=0 \
   --use_raw_cat=0 \
   --p_mask_input1=0.02 \
   --p_mask_input2=0.02 \
   --p_mask_label=0
```

Our best model is saved in "UBARv2-model", which is released at [Google drive](https://drive.google.com/file/d/1dxZ-f08h_4VQZuIWmFD9wNczrd9jlKF1/view?usp=sharing model).



## Evaluation

For a fair comparison, this work conducts experiments and reports results based on the [standardized evaluation scripts](https://github.com/Tomiinek/MultiWOZ_Evaluation ) of MultiWOZ Evaluation .

Our evaluation also implementation supports runing on CPU or a single GPU or multi-GPUs.

### Policy Optimization (Act and Response Generation)

```
load_path='YOUR_EXPERIMENT_PATH'
num_gpus=YOUR_NUM_GPUS
batch_size=YOUR_BATCH_SIZE
input_file_name='inference-test-standard-gen-state.json' 
output_father_folder_path='YOUR_RESULT_SAVING_PATH'
output_file_name='OUTPUT.json'
python -m torch.distributed.launch --nproc_per_node=$num_gpus evaluation.py -mode test -cfg eval_load_path=$load_path batch_size=$YOUR_BATCH_SIZE multi_gpu=True use_true_prev_bspn=True use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=False use_all_previous_context=True

python evaluate.py --bleu --success --input_path=$load_path --input_file_name=$input_file_name --output_father_folder_path=$output_father_folder_path --output_file_name=$output_file_name && echo $input_path is done
```

### End-to-end Modeling (Belief state, Act and Response Generation)

```
load_path='YOUR_EXPERIMENT_PATH'
num_gpus=YOUR_NUM_GPUS
batch_size=YOUR_BATCH_SIZE
input_file_name='inference-test-standard-gen-state.json' 
output_father_folder_path='YOUR_RESULT_SAVING_PATH'
output_file_name='OUTPUT.json'
python -m torch.distributed.launch --nproc_per_node=$num_gpus evaluation.py -mode test -cfg eval_load_path=$load_path batch_size=$YOUR_BATCH_SIZE multi_gpu=True use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=False use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True

python evaluate.py --bleu --success --input_path=$load_path --input_file_name=$input_file_name --output_father_folder_path=$output_father_folder_path --output_file_name=$output_file_name && echo $input_path is done
```

Important note: If you want to evaluate on validation set, you should set input_file_name='inference-validate-standard-gen-state.json'

### Evaluation settings

- use_true_prev_bspn: use the ground truth previous turns' belief span as context.
- use_true_prev_aspn: use the ground truth previous turns' action span as context.
- use_true_db_pointer: use the ground truth database search results as context.
- use_true_prev_resp: use the ground truth previous turns' response as context.
- use_true_curr_bspn: use the ground truth current turn's belief span.
- use_true_curr_aspn: use the ground truth current turn's belief span.
- use_all_previous_context: use all previous turns as context. 
- use_true_bspn_for_ctr_eval: use the ground truth belief span to query DB results.



## Acknowledgement

This code is adapted and modified upon the released code of previous AAAI 2021 paper "UBAR: Towards Fully End-to-End Task-Oriented Dialog Systems with GPT-2".

[UBAR code](https://github.com/TonyNemo/UBAR-MultiWOZ)

[UBAR paper](https://www.aaai.org/AAAI21Papers/AAAI-2262.YangY.pdf )

We appreciate their open-sourcing such high-quality code, which is very helpful to our research.
And of course thanks HuggingFace for their wonderful transformers implementation.