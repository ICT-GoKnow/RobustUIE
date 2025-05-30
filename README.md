# RobustUIE

This is the official code release of the following paper:

Towards Robust Universal Information Extraction: Dataset, Evaluation, and Solution

## Usage

### Environment Setup

```
conda create -n RobustUIE python=3.10

activate RobustUIE

pip install -r requirements.txt
```

### Dataset Preparation

#### 1. Training Data
We adopted the KnowCoder-style data format, which can be found at: [KnowCoder](https://huggingface.co/collections/golaxy/knowcoder-65fc3cd385d98567da412abf).

Note: Because some datasets have copyright requirements and need licenses, we cannot directly release this part of the data now. If you have a license for restricted datasets, you can use them to contact emails in Contact to obtain data.

We conduct experiments utilizing 7 datasets, comprising 3 datasets for the NER task, 2 datasets for the RE task, and 2 datasets for the ED tasks. 

#### 2. RUIE-Bench

We evaluate the robustness of the IE models using our constructed RUIE-Bench. Since some datasets have copyright restrictions and require licenses, we have only released a subset of the data. You can download the data from [Google Drive](https://drive.google.com/file/d/1xCp6hEvEQxXYNe9aunVsOY3Mb2OXC2qi/view?usp=sharing).

### Base Model

We use KnowCoder-7b-base as base model, click [here](https://huggingface.co/golaxy/KnowCoder-7B-base) for download.

### Training
You can run it as follows and you can modify the hyper parametrers in scripts/run.sh.

```
bash scripts/run.sh
```

When the training is over, you need to merge the lora parameters to the original model.

```
bash scripts/merge_lora_ckpts.sh
```

### Loss-guided Data Augmentation

After training a model, you can run the following script to select augmented samples for data augmentation training of the model.

```
python src/LDA_strategy.py
```

### Evaluation
After the training is finished, you can follow the following steps to inference.

```
bash scripts/inference.sh
```

When the inference result is obtained, it can be tested according to the inference result.

```
bash scripts/eval.sh
```

## Citation

If you find the paper helpful, please cite our work:

```
@article{zhu2025towards,
  title={Towards Robust Universal Information Extraction: Benchmark, Evaluation, and Solution},
  author={Zhu, Jizhao and Shi, Akang and Li, Zixuan and Bai, Long and Jin, Xiaolong and Guo, Jiafeng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2503.03201},
  year={2025}
}
