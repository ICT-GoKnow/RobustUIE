import os
import json
import random
# from llmtuner.data import get_dataset
# from llmtuner.hparams import get_train_args
from llmtuner.model import get_train_args
from llmtuner.data import get_dataset, preprocess_dataset
from llmtuner.model import load_model_and_tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    random.seed(42)
    args = None
    
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs,
    )

    # dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    print("*" * 10)
    print(data_args)
    data_args.split = "validation"
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    del model
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    # Predict
    max_num_seqs = 1
    max_model_len = 4096
    max_num_batched_tokens = max_num_seqs * max_model_len
    model_path = model_args.model_name_or_path
    llm = LLM(model=model_path,
                tokenizer_mode='auto',
                trust_remote_code=True,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                seed=42,
                # tensor_parallel_size=4,
                )
    print('***** model loaded *****')

    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=0,
        max_tokens=generating_args.max_new_tokens,
        stop=['</s>'],
    )

    # sampling_params = SamplingParams(
    #     n=1,
    #     best_of=1,
    #     temperature=generating_args.temperature,
    #     top_p=generating_args.top_p,
    #     top_k=generating_args.top_k,
    #     max_tokens=generating_args.max_new_tokens,
    #     stop=['</s>'],
    # )

    outputs = llm.generate(prompt_token_ids=dataset["input_ids"], sampling_params=sampling_params)

    res_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res_list.append(json.dumps({"label": "", "predict": generated_text}, ensure_ascii=False))

    if not os.path.exists(os.path.join("/home/bingxing2/home/scx6592/zyx/tuner-master", f"{training_args.output_dir}")):
        os.makedirs(os.path.join("/home/bingxing2/home/scx6592/zyx/tuner-master", f"{training_args.output_dir}"))

    with open(os.path.join("/home/bingxing2/home/scx6592/zyx/tuner-master", f"{training_args.output_dir}",
                            "generated_predictions.jsonl"), "w", encoding="utf-8") as writer:
        res = []
        writer.write('\n'.join(res_list))
