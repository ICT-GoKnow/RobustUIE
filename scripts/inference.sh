#!/bin/bash

source activate RobustUIE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=1

JOB_ID="${SLURM_JOB_ID}"

date +"%Y-%m-%d %H:%M:%S"

# 可根据实际任务及数据集进行修改
run_tasks=${2:-"NER"}
export CUDA_VISIBLE_DEVICES=0
experiment_names="KnowCoder-original-5epoch"
benchmark_dirs="Conll03/test"
IFS=","
read -r -a array_experiment_names <<< "$experiment_names"
read -r -a array_benchmark_dirs <<< "$benchmark_dirs"
len=${#array_experiment_names[@]}

for ((i=0; i<$len; i++))
do
    experiment_name=${array_experiment_names[$i]}
    benchmark_dir=./benchmark/${array_benchmark_dirs[$i]}

    output_dir=outputs/$experiment_name
    tmp_dir=outputs/$experiment_name
    echo $output_dir
    echo $tmp_dir

    ckpts=${1:-$(ls $tmp_dir| grep 'checkpoint-' | sed -n 's/.*checkpoint-\([0-9]\+\).*/\1/p' | paste -sd, -)}
    echo $ckpts


    log_dir=$output_dir/logs

    if [ ! -d ${log_dir} ];then  
        mkdir ${log_dir}
    fi

    benchmark_name=${benchmark_dir##*/}
    # prediction_dir=$output_dir/lora_merged_results/$benchmark_name
    prediction_dir=$output_dir/lora_merged_results/Conll03/$benchmark_name



    for ckpt in ${ckpts}
    do
        echo "run evaluation for ${ckpt} ..."
        python src/eval_ckpt.py \
            --model_dir $tmp_dir/lora_merged/sft_ckpt_${ckpt} \
            --prompt_corpus_dir $benchmark_dir \
            --run_tasks "$run_tasks" \
            --output_file $prediction_dir/ckpt_${ckpt}_res.pkl \
            --prediction_name "sft_ckpt_${ckpt}" >> $log_dir/eval_${ckpt}_${JOB_ID}.log 2>&1
    done

    date +"%Y-%m-%d %H:%M:%S"

    shell_dir=$output_dir/shells
    if [ ! -d ${shell_dir} ];then  
        mkdir ${shell_dir}
    fi

    cp ${BASH_SOURCE[0]} $shell_dir/eval_ckpts_${JOB_ID}.sh


done
