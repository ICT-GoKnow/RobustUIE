#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@time      : 2023-11-26 00:52:51
@author    : zengyutao1996@163.com
@version   : 1.0
@desc      : 使用vllm进行结果评估, 并存储中间结果
'''

import argparse
import os
import json
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_json_file(file):
    with open(file, 'r', encoding='UTF-8') as file:
        data = json.load(file)
    return data


def read_jsonl_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def dump_json_file(obj, file):
    with open(file, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def dump_jsonl_file(records, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        for record in records:
            outline = json.dumps(record, ensure_ascii=False)
            outfile.write(outline + "\n")


def extract_generated_code(resp,  tokenizer=None, sep='Output:\n',):
    if not isinstance(resp, str):
        import vllm  # noqa: 避免全局依赖vllm

    if isinstance(resp, vllm.outputs.RequestOutput):
        # print(resp.prompt[-200:])
        # print('#'*100)
        if getattr(resp, 'prompt', None) is not None:
            resp = resp.prompt + tokenizer.decode(resp.outputs[0].token_ids)
        else:
            resp = tokenizer.decode(resp.prompt_token_ids + resp.outputs[0].token_ids)
        resp = resp.strip()
    ans = resp.split(sep)
    return ans[-1].strip().replace('</s>', '')


def build_prompt(query, resp='', system=None, sep='\n'):
    if system is None:
        system=(
            "You are a highly skilled assistant at digesting "
            "and extracting information from textual content. "
            "Below is an input containing standard type definitions "
            "and textual content. Please complete it with the "
            "extracted information in the form of structured code."
        )
    #text1="This is an object-oriented programming task: For DATASET CrossNER_AI, some Entity Classes are defined above. Please instantiate all the corresponding Entity Objects in the following sentence from CrossNER_AI."
    #text2="class Entity:\n\t\"\"\"\n\tThe base class for all entities.\n\t\"\"\"\n\tdef __init__(self, name: str):\n\t\tself.name = name\n\nclass AcademicConference(Entity):\n\t\"\"\"\n\tDescription: Refers to a research conference or journal in the field of Artificial Intelligence. This category includes conferences where AI research is presented and published.\n\tExamples: AAAI, 1982 Association for the Advancement of Artificial Intelligence, SIGGRAPH, Symposium on Geometry Processing, International Journal of Computer Vision, IJCV, IEEE Computer Society Conference on Computer Vision and Pattern Recognition, CVPR, International Conference on Machine Learning 2011 & 2012, NIST ' s annual Document Understanding Conferences\n\t\"\"\"\n\tpass\n\nclass Algorithm(Entity):\n\t\"\"\"\n\tDescription: Refers to an algorithmic procedure or computational model used in Artificial Intelligence research. This category includes algorithms (e.g., decision trees) and models (e.g., CNN and LSTM).\n\tExamples: principal component analysis, linear discriminant analysis, gradient descent, Support vector machine, recurrent neural network, LSTM, PCA, LDA, canonical correlation analysis, CCA\n\t\"\"\"\n\tpass\n\nclass BranchOfScience(Entity):\n\t\"\"\"\n\tDescription: Refers to a specific research area or subfield within Artificial Intelligence. Also annotate acronyms such as NLP.\n\tExamples: deep learning, pattern recognition, image processing, reinforcement learning, natural language processing, machine learning, unsupervised learning, AI, computer vision, text mining\n\t\"\"\"\n\tpass\n\nclass Country(Entity):\n\t\"\"\"\n\tDescription: Refers to a sovereign nation.\n\tExamples: Netherlands, Japan, Germany, Canada, Australia, Brazil, China, India, Italy, Korea\n\t\"\"\"\n\tpass\n\nclass Human(Entity):\n\t\"\"\"\n\tDescription: Refers to an individual's name that is not a researcher.\n\tExamples: Francis Ford Coppola, Michael Jackson, John Wayne, Rita Hayworth, Dean Martin, Jerry Lewis\n\t\"\"\"\n\tpass\n\nclass Metrics(Entity):\n\t\"\"\"\n\tDescription: Refers to evaluation metrics used to assess the performance of AI models and algorithms. Annotate specific metrics like F1-score.\n\tExamples: mean squared error, DCG, maximum likelihood, Recall-Oriented Understudy for Gisting Evaluation, MSE, noise floor measurement, ROUGE, Hinge loss, hinge loss, Sigmoid function Cross entropy loss\n\t\"\"\"\n\tpass\n\nclass Organization(Entity):\n\t\"\"\"\n\tDescription: Refers to a structured group, institution, company, or association. This category covers a diverse range of organizations, including businesses, non-profits, educational institutions, and government agencies.\n\tExamples: Unimation, IAPR, Audio Engineering Society, National Science Foundation, National Aeronautics and Space Administration, NASA, US Department of Energy, US Department of Commerce NIST, US Department of Defense, Defense Advanced Research Projects Agency\n\t\"\"\"\n\tpass\n\nclass Other(Entity):\n\t\"\"\"\n\tDescription: Refers to named entities that are not included in any other category.\n\tExamples: unsupervised methods, topological properties, audio signal, eigenfaces, intelligent agents, graphical user interfaces, Heuretics : Theoretical and Study of Heuristic Rules, Best Paper award, Johann Bernoulli Chair, Toshiba Endowed Chair\n\t\"\"\"\n\tpass\n\nclass Product(Entity):\n\t\"\"\"\n\tDescription: Refers to a product, system, or toolkit related to Artificial Intelligence. This includes specific AI-enabled products (e.g., robots like Pepper), systems (e.g., facial recognition systems), and toolkits (e.g., Tensorflow and PyTorch).\n\tExamples: MATLAB, Programmable Universal Machine for Assembly, industrial robot, opinion-based recommender system, Octave, Google Translate, SYSTRAN system, BabelFish, Babelfish, RapidMiner\n\t\"\"\"\n\tpass\n\nclass ProgrammingLanguage(Entity):\n\t\"\"\"\n\tDescription: Refers to a programming language used in the development of AI applications and research. Annotate the name of the programming language, such as Java and Python.\n\tExamples: Java, R, CLIPS, Python, C + +, GNU Octave, Java 9, Perl, ActiveX, .NET\n\t\"\"\"\n\tpass\n\nclass Researcher(Entity):\n\t\"\"\"\n\tDescription: Refers to an individual engaged in research activities within the field of Artificial Intelligence (AI), including professors, Ph.D. students, and researchers in academia, research institutions, and companies. If a person is involved in AI research, label them as a researcher entity instead of a person entity.\n\tExamples: Jürgen Schmidhuber, Seymour Papert, Victor Scheinman, Scheinman, X.Y. Feng, H. Zhang, Y.J. Ren, P.H. Shang, Y. Zhu, Y.C. Liang\n\t\"\"\"\n\tpass\n\nclass SpatialEntity(Entity):\n\t\"\"\"\n\tDescription: Refers to a specific geographical or structural location. This includes but is not limited to: places (e.g., parks, landmarks), bridges, cities, towns, villages, areas and other distinct regions.\n\tExamples: Chișinău, Paris, Montreal, Scotiabank Theatre Toronto, TIFF Bell Lightbox, Moldavian SSR\n\t\"\"\"\n\tpass\n\nclass Task(Entity):\n\t\"\"\"\n\tDescription: Refers to a particular research task or problem within a specific AI research field. Annotate the name of the specific task, such as machine translation or object detection.\n\tExamples: speech synthesis, information retrieval, Feature extraction, dimension reduction, speech recognition, sentiment analysis, Multimodal sentiment analysis, face recognition, handwriting recognition, lip reading\n\t\"\"\"\n\tpass\n\nclass University(Entity):\n\t\"\"\"\n\tDescription: Refers to an educational institution of higher learning. Label organizations that are universities as 'university'' entities.\n\tExamples: University of Toronto, Cambridge, University of Groningen, MIT, Brown University, Carnegie Mellon University, MPI Saarbruecken, Stanford University, University of California , San Diego, École Centrale Paris\n\t\"\"\"\n\tpass\n\n\"\"\"\n"
    ##text2="List all the Entity words in the following sentence as instances of corresponding subclasses of class Entity. If there do not exist any Entity words that belong to the Entity subclasses we defined, print \"None\".\n"
    ##text3="Extract the entities from the following sentence.\n"
    #text4="Some entity types are given above, please find all the entities in the above types in the sentence."
    #text5="Some Classes are defined above, please instantiate the Objects corresponding to the above Classes in the sentence."
    #query=query.replace("from Entities import ChemicalSubstance, Periodical, Specialty, GroupOfHumans, LegalPerson, HigherEducationInstitution, Occurrence, Person, ChemicalCompound, PhysicalObject, ArtificialEntity, Idea, PoliticalTerritorialEntity, Part\n\n","")
    ##movie
    #query=query.replace("from Entities import IntellectualWork, Evaluation, LegalPerson, Class, TimeInterval, FictionalCharacter, Poetry, Identifier, Manager, LiteraryWork, CulturalHeritage\n\n","")
    ##query+="\nfrom Entities import AcademicConference,Algorithm,BranchOfScience,Country,Human,Metrics,Organization,Other,Product,ProgrammingLanguage,Researcher,SpatialEntity,Task,University"
    ##liture
    #query=query.replace("from Entities import ArtGenre, Periodical, GroupOfHumans, Occurrence, Person, Author, WrittenWork, ArtificialEntity, LiteraryWork, PoliticalTerritorialEntity\n\n","")
    ##rest
    #query=query.replace("from Entities import Evaluation, Measure, TimeInterval, Service, Product, Culture, Identifier\n\n","")
    ##poli
    #query=query.replace("from Entities import GroupOfHumans, LegalPerson, Occurrence, Person, Voting, Organization, PoliticalTerritorialEntity\n\n","")
    #query=query.replace("from Entities import Group, ArtGenre, LegalPerson, Tool, GroupOfHumans, Occurrence, Person, Poetry, ArtificialEntity, PoliticalTerritorialEntity, MusicalEnsemble\n\n","")
    ##query=query.replace("from Entities import ArtGenre, Periodical, GroupOfHumans, Occurrence, Person, Author, WrittenWork, ArtificialEntity, LiteraryWork, PoliticalTerritorialEntity\n\n","")
    ##print(query)
    ###print(query)
    ##ai
    #query=query.replace("from Entities import Specialty, Convention, GroupOfHumans, ComputerLanguage, LegalPerson, HigherEducationInstitution, Person, Activity, ArtificialEntity, PoliticalTerritorialEntity, Work\n\n","")
    #print(query)
    # pattern = "Input:\n{query}\nOutput:\n{resp}"
    pattern = "[INST] {instruction} [/INST]"
    # pattern = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    if system:
        return pattern.format(instruction=system + '\n' + query)
    else:
        return pattern.format(instruction=query)


def build_input(tokenizer, query, resp='', system=None, sep='\n', head_max_len=1500, tail_max_len=2000):
    prompt = build_prompt(query=query, resp=resp, system=system, sep=sep)
    token_ids = tokenizer.encode(prompt)
    if head_max_len is not None and tail_max_len is not None and head_max_len + tail_max_len < len(token_ids):
        return token_ids[:head_max_len] + token_ids[-tail_max_len:]
    return token_ids


def retrieve_dict(dic, key_with_sep, sep='.'):
    level_key = key_with_sep.split(sep)
    val = dic
    for key in level_key:
        val = val[key]
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, help='model dir')
    parser.add_argument('--prompt_corpus_dir', type=str,
                        default='/workspace/user_code/work/factgpt/corpus/prompt_construction',
                        help='the dir to store prompt corpus')
    parser.add_argument('--run_tasks', type=str, default='EE,NER,RE', help='tasks to run')
    parser.add_argument('--output_file', type=str, default=None, help='output file')
    parser.add_argument('--prediction_name', type=str, default='prediction', help='prediction name')
    args = parser.parse_args()

    if args.model_dir is None:
        raise ValueError('model_dir is None')

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    args.run_tasks = args.run_tasks.split(',')

    # 1. 准备模型
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,  # llama的tokenizer似乎存在问题(使用fast的同时需要设置legacy为False)
        legacy=False,
        padding_side='left',  # training时padding在右侧, generation时padding在左侧
        trust_remote_code=True,
    )
    # model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    # 2. 准备数据
    test_task_files = {
        'ED': f'{args.prompt_corpus_dir}/ED/test-prompt.json',
        'NER': f'{args.prompt_corpus_dir}/NER/test-prompt.json',
        'RE': f'{args.prompt_corpus_dir}/RE/test-prompt.json',
        'EAE': f'{args.prompt_corpus_dir}/EAE/test-prompt.json',
        'ED-1dataset': f'{args.prompt_corpus_dir}/ED-1dataset/test-prompt.json',
        'EAE-1dataset': f'{args.prompt_corpus_dir}/EAE-1dataset/test-prompt.json',
        'EE': f'{args.prompt_corpus_dir}/EE/test-prompt.json',
    }

    # 2.1. 不同的数据设置
    query_settings = [
        # 'one-stage.zero-shot.prompt--1500_2000',
        'one-stage.first-5-shot.prompt--1500_2000',
    ]

    all_prompts = {}
    for task_type in args.run_tasks:
        test_data_file = test_task_files[task_type]
        all_prompts[task_type] = {}
        test_data = read_jsonl_file(test_data_file)
        for it in tqdm(test_data):
            for setting in query_settings:
                if setting not in all_prompts[task_type]:
                    all_prompts[task_type][setting] = []
                if '--' in setting:
                    key, _max_len = setting.split('--')
                    head_max_len, tail_max_len = [int(x) for x in _max_len.split('_')]
                else:
                    key, head_max_len, tail_max_len = setting, None, None
                query = retrieve_dict(it, key)
                token_ids = build_input(tokenizer=tokenizer, query=query,
                                        head_max_len=head_max_len, tail_max_len=tail_max_len)
                all_prompts[task_type][setting].append(token_ids)

    max_num_seqs = 4
    max_model_len = 4096
    max_num_batched_tokens = max_num_seqs * max_model_len
    llm = LLM(model=args.model_dir,
              tokenizer_mode='auto',
              trust_remote_code=True,
              max_num_seqs=max_num_seqs,
              max_model_len=max_model_len,
              max_num_batched_tokens=max_num_batched_tokens)
    print('model loaded!!!!')

    # 3. 准备解码方式
    # greedy_search设置方式
    max_out_len = 512 + 128  # 从benckmark中统计的，后续需要的话再调整
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=0.75,
        stop='"""',
        max_tokens=max_out_len,
        top_p=0.9,
    )

    # 4. 生成结果
    print(f'run_tasks: {args.run_tasks}')
    output_res = {}
    for task_type in args.run_tasks:
        output_res[task_type] = {}
        for setting in query_settings:
            print(f'processing: {task_type}-{setting} ...')
            output_res[task_type][setting] = [''] * len(all_prompts[task_type][setting])
            good_indices = [idx for idx, seq in enumerate(all_prompts[task_type][setting]) if len(seq) < max_model_len]
            good_len, bad_len = len(good_indices), len(all_prompts[task_type][setting]) - len(good_indices)
            print(f'good_samples={good_len}, bad_samples={bad_len}, all_samples={len(all_prompts[task_type][setting])}')
            
            good_samples = [all_prompts[task_type][setting][idx] for idx in good_indices]
            # print(good_samples[:3])
            good_sample_results = llm.generate(prompt_token_ids=good_samples, sampling_params=sampling_params)

            assert len(good_indices) == len(good_sample_results)
            for idx, r in zip(good_indices, good_sample_results):
                output_res[task_type][setting][idx] = r

    # 缓存中间生成结果
    with open(args.output_file, 'wb') as file:
        pickle.dump(output_res, file)

    # 5. 提取生成结果至中间文件
    for task_type in args.run_tasks:
        # 抽取步骤较为耗时, 需要前置, 避免出现读取数据不同步
        task_res = {}
        for setting in query_settings:
            setting_res = [extract_generated_code(r, tokenizer) for r in output_res[task_type][setting]]
            task_res[setting] = setting_res

        assert len(set([len(v) for v in task_res.values()])) == 1, '不同setting下结果长度应该相同'

        # 中间文件名: intermediate_{task_type}.json
        mid_file = f'{output_dir}/intermediate_{task_type}.json'
        if os.path.exists(mid_file):
            cur_data = read_json_file(mid_file)
        else:
            cur_data = read_jsonl_file(test_task_files[task_type])

        assert len(cur_data) == len(task_res[query_settings[0]]), '运行样本数应该相同'

        for idx in tqdm(range(len(cur_data))):
            for setting in query_settings:
                if "[/INST]" in task_res[setting][idx]:
                    cur_data[idx][args.prediction_name + '-' + setting] = task_res[setting][idx].split("[/INST]")[1]
                if "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" in task_res[setting][idx]:
                    cur_data[idx][args.prediction_name + '-' + setting] = task_res[setting][idx].split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
                else:
                    cur_data[idx][args.prediction_name + '-' + setting] = task_res[setting][idx]

        dump_json_file(cur_data, mid_file)
        print(f'add field-{args.prediction_name} to {mid_file}')
