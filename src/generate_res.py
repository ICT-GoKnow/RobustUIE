#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
import json
from utils import read_jsonl_file,read_json_file, dump_json_file, extract_generated_code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, help='model dir')
    parser.add_argument('--prompt_corpus_dir', type=str,
                        default='/workspace/user_code/work/factgpt/corpus/prompt_construction',
                        help='the dir to store prompt corpus')
    parser.add_argument('--run_tasks', type=str, default='EAE,NER,RE', help='tasks to run')
    parser.add_argument('--output_file', type=str, default=None, help='output file')
    parser.add_argument('--prediction_name', type=str, default='prediction', help='prediction name')
    parser.add_argument('--shot', type=int, required=True, choices=[0,5], help='shot')
    args = parser.parse_args()

    if args.shot == 0:
        prompt_key = 'zero-shot'
    elif args.shot == 5:
        prompt_key = 'top-5-shot'
    
    # refine prediction name
    args.prediction_name = f'{args.prediction_name}.{prompt_key}'
    # add the model name to the prediction name
    args.prediction_name = f'{args.model_dir.split("/")[-1]}.{args.prediction_name}' 
    
    print(f'prediction_name: {args.prediction_name}')
    
    
    if args.model_dir is None:
        raise ValueError('model_dir is None')

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    args.run_tasks = args.run_tasks.split(',')

    # 1. 准备模型
    # location = 'gz'
    # location = 'cq'
    # model_dir = f'/cfs/cfs-{location}/save/pretrained_models/FactGPT/sft_corpus_pt_llama2_13b_1024_ckpt_1000'
    # model_dir = '/cfs/cfs-hcxu53n5/pedroyang/outputs/llm/llama2_13b_wiki_2048/global_step200_hf'
    print(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,  # llama的tokenizer似乎存在问题(使用fast的同时需要设置legacy为False)
        legacy=False,
        padding_side='left',  # training时padding在右侧, generation时padding在左侧
        trust_remote_code=True,
    )
    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    max_num_batched_tokens = 16000
    llm = LLM(model=args.model_dir,
              tokenizer_mode='auto',
              trust_remote_code=True,
              max_num_batched_tokens=max_num_batched_tokens)
    
    print('model loaded!!!!')

    # 2. 准备数据
    test_task_files = {
        #'EAE': f'{args.prompt_corpus_dir}/EAE/test-prompt.json',
        'EE': f'{args.prompt_corpus_dir}/EE/test-prompt.json',
        'NER': f'{args.prompt_corpus_dir}/NER/test-prompt.json',
        'RE': f'{args.prompt_corpus_dir}/RE/test-prompt.json',
    }

    all_prompts = {}
    for task_type, test_data_file in test_task_files.items():
        all_prompts[task_type] = []
        test_data = read_jsonl_file(test_data_file)
        for it in tqdm(test_data):
            #print(it)
            it=it['one-stage'][prompt_key]
            key = 'prompt' if 'prompt' in it else 'prompt_1'
            #key='prompt'
            
            
            prompt = it[key]
            
            all_prompts[task_type].append(prompt)

    print(all_prompts['NER'][:3])
    # task_types = {
    #     'EAE': {'prompt': str},
    #     'EE': {'prompt_1': str, 'prompt_2': {'stage_1': str, 'stage_2': str}},
    #     'NER': {'prompt_1': str, 'prompt_2': {'stage_1': str, 'stage_2': str}},
    #     'RE': {'prompt_1': str, 'prompt_2': {'stage_1': str, 'stage_2': str}},
    # }

    # 3. 准备解码方式
    # greedy_search设置方式
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=0,
        stop='"""',
        max_tokens=512,
    )

    # 4. 生成结果
    print(f'run_tasks: {args.run_tasks}')
    output_res = {}
    for task_type in args.run_tasks:
        #print(all_prompts[task_type])
        output_res[task_type] = llm.generate(all_prompts[task_type], sampling_params)

    # 缓存中间生成结果
    with open(args.output_file, 'wb') as file:
        pickle.dump(output_res, file)

    # 5. 提取生成结果至中间文件
    for task_type in args.run_tasks:
        # 中间文件名: intermediate_{task_type}.json
        mid_file = f'{output_dir}/intermediate_{task_type}.json'
        if os.path.exists(mid_file):
            cur_data = read_json_file(mid_file)
        else:
            cur_data = read_jsonl_file(test_task_files[task_type])

        for idx in tqdm(range(len(cur_data))):
            cur_data[idx][args.prediction_name] = extract_generated_code(output_res[task_type][idx], tokenizer)

        dump_json_file(cur_data, mid_file)
        print(f'add field-{args.prediction_name} to {mid_file}')
