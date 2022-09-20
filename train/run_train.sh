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