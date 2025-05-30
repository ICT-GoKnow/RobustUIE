import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def build_prompt(query, resp='', system=None, sep='\n'):
    if system is None:
        system = (
            "You are a highly skilled assistant at digesting "
            "and extracting information from textual content. "
            "Below is an input containing standard type definitions "
            "and textual content. Please complete it with the "
            "extracted information in the form of structured code."
        )
    pattern = "Input:\n{query}\nOutput:\n{resp}"
    if system:
        return sep.join([system, pattern.format(query=query, resp=resp)])
    else:
        return pattern.format(query=query, resp=resp)


def build_input(tokenizer, query, resp, system=None, sep='\n', head_max_len=1500, tail_max_len=2000):
    prompt = build_prompt(query=query, resp=resp, system=system, sep=sep)
    token_ids = tokenizer.encode(prompt)
    if head_max_len is not None and tail_max_len is not None and head_max_len + tail_max_len < len(token_ids):
        token_ids = token_ids[:head_max_len] + token_ids[-tail_max_len:]
    return token_ids


def calculate_loss(data, tokenizer, model, device, head_max_len=1500, tail_max_len=2000):
    """
    Calculate the loss for each sample and return a list of samples with loss.

    Parameters:
    ner_data (list): List of input data, each element contains a dictionary containing 'one-stage' and 'output' information.
    tokenizer (Tokenizer): The tokenizer used to encode the text.
    model (torch.nn.Module): The model used to calculate the loss.
    device (torch.device): The computing device of the model, usually 'cuda' or 'cpu'.
    head_max_len (int): The maximum length of the input text head, default is 1500.
    tail_max_len (int): The maximum length of the input text tail, default is 2000.

    Returns:
    list: A list of samples after calculating the loss, each sample contains the loss value.
    """
    samples_with_loss = []

    for item in data:
        with torch.no_grad():
            input_text = str(item['one-stage']['zero-shot']['prompt'])
            target_text = str(item['one-stage']['output'])

            # Encoding input text
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)

            # Encoding target text
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            # Construct the complete input sequence
            full_input_ids = build_input(tokenizer, input_text, target_text, head_max_len=head_max_len,
                                         tail_max_len=tail_max_len)

            # Convert full_input_ids to a PyTorch tensor and move it to the GPU
            full_input_ids = torch.tensor([full_input_ids]).to(device)

            # Create labels, the target text part is target_ids, and the rest is -100 (ignore the loss)
            labels = torch.full_like(full_input_ids, -100).to(device)
            labels[0, -len(target_ids):] = torch.tensor(target_ids).to(device)

            # Model output
            outputs = model(input_ids=full_input_ids, labels=labels)
            loss = outputs.loss.item()

            # Output loss
            print(f"Inference loss for sample: {loss}")

            item['loss'] = loss
            samples_with_loss.append(item)

    return samples_with_loss


def filter_data_by_loss(data, dataset_name, perturbation_type, top_percentage):
    """
    Filter data based on dataset name and noise type, and return high loss samples.
    """
    filtered_data = [item for item in data if item['dataset'] == dataset_name]
    filtered_data.sort(key=lambda x: x['loss'], reverse=True)
    top_n = int(len(filtered_data) * top_percentage)
    return filtered_data[:top_n]


def run_loss_based_filtering(args, configurations):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(device)
    model.eval()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    samples_with_loss = calculate_loss(
        data, tokenizer, model, device,
        head_max_len=args.head_max_len,
        tail_max_len=args.tail_max_len
    )

    final_samples = []
    for dataset_name, perturbation_type in configurations:
        filtered = filter_data_by_loss(samples_with_loss, dataset_name, perturbation_type, args.top_percentage)
        final_samples.extend(filtered)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in final_samples:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KnowCoder')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--top_percentage', type=float, default=0.1)
    parser.add_argument('--head_max_len', type=int, default=1500)
    parser.add_argument('--tail_max_len', type=int, default=2000)
    args = parser.parse_args()

    # Datasets and perturbation types can be modified or added
    configurations = [
        ('Conll2003', 'Replace Entity'),
        ('Conll2003', 'Mask Context'),
        ('Conll2003', 'Extend Sentence'),
        ('Conll2003', 'Typo Injection'),
        ('Conll2003', 'Lowercase Conversion'),
    ]

    run_loss_based_filtering(args, configurations)
