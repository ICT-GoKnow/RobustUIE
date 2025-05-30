#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export OMP_NUM_THREADS=32

source activate RobustUIE
export WANDB_DISABLED=true

dataset=Original_test

# 根据不同的训练数据修改输出路径

output_dir=outputs/KnowCoder-original-5epoch

curDate=$(date "+%Y%m%d%H%M%S")
ds_config=configs/deepspeed/ds_config_2.json

OUTPUT_LOG="logs/train_${dataset}_${curDate}.log"
model_name_or_path=KnowCoder-7b-base

template=KnowCoder

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnodes 1 \
    --nproc_per_node 4 \
    --node_rank=0 \
    --master_port 16666 \
    src/train_bash.py \
    --deepspeed ${ds_config} \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --lora_target all \
    --lora_rank 32 \
    --model_name_or_path ${model_name_or_path} \
    --template ${template} \
    --dataset ${dataset} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 1000 \
    --learning_rate 3e-4 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.1 \
    --plot_loss \
    --fp16 \
    --flash_attn \
    --seed 42 \
    --ddp_timeout 1800000 \
    --dataloader_num_workers 1 \
    --cutoff_len 2048 >> $OUTPUT_LOG 2>&1 \