
source activate RobustUIE


# <need change>
data_dir="outputs/KnowCoder-original-5epoch/lora_merged_results/Conll03/test"
# 评估结果输出目录
tmp_dir="outputs/KnowCoder-original-5epoch/lora_merged_results/Conll03/test"
# <\need change>


# 可视化结果输出目录
visual_dir="tmp/1126_test_with_filter_demo2"
# 评估结果输出目录 (用于单图多线绘制模式，提供多个数据源目录)
tmp_dirs=${1:-"tmp/1126_test,tmp/1126_test_with_filter,tmp/1126_test_without_filter"}
# 曲线名称 (用于单图多线绘制模式，提供多个数据源曲线名称)
name_list=${2:-"skip+no-filter,no-skip+filter,no-skip+no-filter"}
# 本体数据目录
ontology_dir="/home/bingxing2/home/scx6592/corpus/onto_data"
# 匹配类型
eval_type="file"   # choices = {'file': '生成中间文件 prediction.json 和 label.json', 'obj': '不生成中间文件'}
# Python 环境
python_env=python


# 需要评估的 ckpts
# <need change>
ckpts=${1:-"1000,2000"}



# 评估任务
# tasks=${2:-"NER","RE","EAE","ED"}
tasks=${2:-"NER"}

# 实验设置
# experiment_sets=${6:-"prompt--1500_2000"}
experiment_sets=${6:-"one-stage.zero-shot.prompt--1500_2000"}
# prompt--1500_2000
# one-stage.zero-shot.prompt--1500_2000

# <\need change>

# bash scripts/eval_ckpts_res2_zhKnowCoder_ie.sh

# 匹配类型
match_types=${3:-"EM"}
# 评估粒度
eval_granularities=${4:-"overall,source,type"}
# 基础模型
#base_models=${5:-"knowcoder_v0.1,llama2"}
# 是否过滤
filter_outlier=0 # choices = {0: '不过滤', '1': '过滤'}
# 是否进行 wikidata 上位转换
wikidata_upper=0 # choices = {0: '不转换', '1': '转换'}
# schema 版本
schema_type="aligned"   # choices = {'aligned': '本体对齐版本 schema', 'unaligned': '未本体对齐版本 schema'}
# 是否汇总指标
summary=1   # choices = {0: '不汇总', '1': '汇总'}
# 是否可视化指标
visualization=0   # choices = {0: '不可视化', '1': '可视化'}
# 绘图模式
visual_mode="single"   # choices = {'single': '绘制单图单线', 'multi': '绘制单图多线'}
# 定义映射
declare -A name_map
name_map=([EAE]="Event" [RE]="Relation" [NER]="Entity" [EE]="Event")
IFS=","


# 评估结果，计算指标

for experiment_set in ${experiment_sets}
do
    for ckpt in ${ckpts}
    do
        if [ $eval_type = 'file' ];then
            echo "start to convert checkpoint-${ckpt} result!"
            if [[ ${tasks} =~ "EAE" ]]; then
                ${python_env} src/convert/convert_eae.py \
                    --input_file ${data_dir}/intermediate_EAE.json \
                    --output_dir ${tmp_dir}/${experiment_set}/EAE/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "NER" ]]; then
                ${python_env} src/convert/convert_ner.py \
                    --input_file ${data_dir}/intermediate_NER.json \
                    --output_dir  ${tmp_dir}/${experiment_set}/NER/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --wikidata_upper ${wikidata_upper} \
                    --filter_outlier ${filter_outlier} \
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "RE" ]]; then
                ${python_env} src/convert/convert_re.py \
                    --input_file ${data_dir}/intermediate_RE.json \
                    --output_dir ${tmp_dir}/${experiment_set}/RE/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            if [[ ${tasks} =~ "ED" ]]; then
                ${python_env} src/convert/convert_ed.py \
                    --input_file ${data_dir}/intermediate_ED.json \
                    --output_dir ${tmp_dir}/${experiment_set}/ED/${ckpt} \
                    --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                    --filter_outlier ${filter_outlier}\
                    --ontology_dir ${ontology_dir} \
                    --schema_type ${schema_type}
            fi
            echo "start to evaluate checkpoint-${ckpt} result!"
            for task_type in ${tasks}
            do
                for match_type in ${match_types}
                do
                    for eval_granularity in ${eval_granularities}
                    do
                        echo "start to evaluate ${task_type} with ${match_type} and ${eval_granularity}!"
                        ${python_env} src/eval.py \
                            --input_file ${data_dir}/intermediate_${task_type}.json \
                            --pred_file ${tmp_dir}/${experiment_set}/${task_type}/${ckpt}/prediction.json \
                            --gold_file ${tmp_dir}/${experiment_set}/${task_type}/${ckpt}/label.json \
                            --task_type ${task_type} \
                            --match_type ${match_type} \
                            --output_dir ${tmp_dir}/${experiment_set}/${task_type}/${ckpt} \
                            --result_file result_${eval_granularity}_${match_type}.json \
                            --granularity ${eval_granularity}
                    done
                done
            done
        else
            echo "start to evaluate checkpoint-${ckpt} result!"
            for task_type in ${tasks}
            do
                for match_type in ${match_types}
                do
                    for eval_granularity in ${eval_granularities}
                    do
                        echo "start to evaluate ${task_type} with ${match_type} and ${eval_granularity}!"
                        ${python_env} src/eval.py \
                            --input_file ${data_dir}/intermediate_${task_type}.json \
                            --prediction_name sft_ckpt_${ckpt}-${experiment_set} \
                            --eval_type ${eval_type} \
                            --task_type ${task_type} \
                            --match_type ${match_type} \
                            --output_dir ${tmp_dir}/${experiment_set}/${task_type}/${ckpt} \
                            --result_file result_${eval_granularity}_${match_type}.json \
                            --granularity ${eval_granularity} \
                            --wikidata_upper ${wikidata_upper} \
                            --filter_outlier ${filter_outlier}\
                            --ontology_dir ${ontology_dir} \
                            --schema_type ${schema_type}
                    done
                done
            done
        fi
    done
done



# 汇总指标
if [ $summary = 1 ]; then
    
    for experiment_set in ${experiment_sets}
    do
        ${python_env} src/combine_eval_results.py \
            --tasks "${tasks}" \
            --match_types "${match_types}" \
            --ckpts "${ckpts}" \
            --data_dir "${tmp_dir}/${experiment_set}" \
            --output_dir "${tmp_dir}/${experiment_set}"
    done
    
fi


# 可视化指标
if [ $visualization = 1 ]; then
    if [ $visual_mode = 'single' ]; then
        for base_model in ${base_models}
        do
            for experiment_set in ${experiment_sets}
            do
                ${python_env} src/visualization_single.py \
                    --tasks "${tasks}" \
                    --ckpts "${ckpts}" \
                    --data_dir "${tmp_dir}/${base_model}/${experiment_set}" \
                    --output_dir "${visual_dir}/${base_model}/${experiment_set}" \
                    --save_visul_results
            done
        done
    else
        for base_model in ${base_models}
        do
            for experiment_set in ${experiment_sets}
            do
                ${python_env} src/visualization_multi.py \
                    --tasks "${tasks}" \
                    --ckpts "${ckpts}" \
                    --data_dirs "${tmp_dirs}" \
                    --sub_dir "${base_model}/${experiment_set}" \
                    --name_list "${name_list}" \
                    --output_dir "${visual_dir}/${base_model}/${experiment_set}" \
                    --save_visul_results
            done
        done
    fi
fi
tmp_dir="/home/bingxing2/home/scx6592/llama_training2/outputs/llm/llama2_7b_20240130_kelm_ner_v7_single_remove-hf/lora_merged_results/sft2_aligned_uie_prompt_7datasets_v15_fewshot/one-stage.first-5-shot.prompt--1500_2000"
# 导出最终对比指标的目录
output_dir="/home/bingxing2/home/scx6592/llama_training2/outputs/llm/llama2_7b_20240130_kelm_ner_v7_single_remove-hf/lora_merged_results/sft2_aligned_uie_prompt_7datasets_v15_fewshot/one-stage.first-5-shot.prompt--1500_2000"
# 需要评估的checkpoints
# python环境
python_command=python
# 需要评估的任务
# 匹配的类型

eval_granularities==${8:-"type,source,overall"}

# 切换分隔符为逗号
IFS=","
for eval_granular in ${eval_granularities}
do
    ${python_command} src/visualization.py  \
        --run_tasks "${run_tasks}" \
        --run_match_types "${run_match_typs}" \
        --ckpts "${ckpts}" \
        --intermediate_data_dir "${tmp_dir}" \
        --output_dir "${output_dir}" \
        --granularity "${eval_granular}"
done

