#!/bin/bash


source activate RobustUIE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=1

JOB_ID="${SLURM_JOB_ID}"

date +"%Y-%m-%d %H:%M:%S"


# 注意：修改为前序checkpoint的路径
experiment_names="KnowCoder-original-5epoch"
model_name_or_path=KnowCoder-7b-base
template=KnowCoder
IFS=","

for experiment_name in $experiment_names
do
    echo $experiment_name
    output_dir=outputs/$experiment_name
    tmp_dir=outputs/$experiment_name
    ckpts=${1:-$(ls $tmp_dir| grep 'checkpoint-' | sed -n 's/.*checkpoint-\([0-9]\+\).*/\1/p' | paste -sd, -)}
    echo $ckpts
    log_dir=$output_dir/logs
    if [ ! -d ${output_dir} ];then  
        mkdir ${output_dir}
    fi

    if [ ! -d ${log_dir} ];then  
        mkdir ${log_dir}
    fi

    for ckpt in ${ckpts}
    do
        python src/export_model.py \
            --model_name_or_path $model_name_or_path \
            --template $template \
            --finetuning_type lora \
            --checkpoint_dir $tmp_dir/checkpoint-${ckpt} \
            --export_dir $tmp_dir/lora_merged/sft_ckpt_${ckpt} >> $log_dir/merge_${ckpt}_${JOB_ID}.log 2>&1
    done
done
# --export_dir $output_dir/lora_merged/sft_ckpt_${ckpt} >> $log_dir/merge_${ckpt}_${JOB_ID}.log 2>&1

date +"%Y-%m-%d %H:%M:%S"

shell_dir=$output_dir/shells
if [ ! -d ${shell_dir} ];then  
    mkdir ${shell_dir}
fi

cp ${BASH_SOURCE[0]} $shell_dir/merge_lora_ckpts_${JOB_ID}.sh
