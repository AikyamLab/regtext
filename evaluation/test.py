from unsloth import FastLanguageModel
import pandas as pd
import os
from datasets import Dataset
from parser_ import get_args
from compute_metrics import compute_metrics, compute_grouped_metrics
import json
from tqdm import tqdm
import random
from huggingface_hub import login

login(token=args.hf_token)
args = get_args()


# fixed parameters
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

data_mode = args.data_mode

# text,label,category,task
file_path = args.file_path
test_file_path = args.test_file_path
assert os.path.isdir(file_path)
train_file_path = os.path.join(file_path, "train.csv")
test_file_path = os.path.join(test_file_path, "test.csv")

model_name = args.model_name

path = args.save_dir

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = os.path.join(args.ckpt_dir, f"{model_name}_{data_mode}_lora_model"),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
except:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = os.path.join("unsloth", args.model_name),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

alpaca_prompt_default = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

alpaca_prompt_polarity = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Answer in not more than 2 words.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    # for instruction, input, output in zip(instructions, inputs, outputs):
    # Must add EOS_TOKEN, otherwise your generation will go on forever!
    inputs, outputs, instruction = examples['text'], examples['label'], examples['definition']
    if "polarity" in args.save_dir:
        text = alpaca_prompt_polarity.format(instruction, inputs, outputs) + EOS_TOKEN
    else:
        text = alpaca_prompt_default.format(instruction, inputs, outputs) + EOS_TOKEN

    # import ipdb;ipdb.set_trace()
    return text

##################################### Train, Test #######################################################
dataset_train = pd.read_csv(train_file_path).fillna("None")
train = dataset_train.copy()
train['input'] = dataset_train['text']

for i in range(dataset_train.shape[0]):
    # import ipdb;ipdb.set_trace()
    train.loc[i, 'input'] = formatting_prompts_func(train.iloc[i])

dataset_test = pd.read_csv(test_file_path).fillna("None")
test = dataset_test.copy()
test['input'] = dataset_test['text']

for i in range(dataset_test.shape[0]):
    test.loc[i, 'input'] = formatting_prompts_func(test.iloc[i])

train_dataset = Dataset.from_dict({
    "input": [str(item) for item in train['input']], 
    "instruction": [str(item) for item in train['definition']], 
    'output': [str(item) for item in train['label']], 
    'text': [str(item) for item in train['text']],
    "category": [str(item) for item in train['category']],
    "task": [str(item) for item in train["task"]]
})

test_dataset = Dataset.from_dict({
    "input": [str(item) for item in test['input']], 
    "instruction": [str(item) for item in test['definition']], 
    'output': [str(item) for item in test['label']], 
    'text': [str(item) for item in test['text']],
    "category": [str(item) for item in test['category']],
    "task": [str(item) for item in test["task"]]
})


######################################################################################################################=
train_pred_label = []
train_orig_label = []
test_pred_label = []
test_orig_label = []
test_categories, train_categories = [], []
test_tasks, train_tasks = [], []
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


end_token = EOS_TOKEN

# get 100 random indices for train from each task
train_indices = random.sample(list(range(train_dataset.shape[0])), train_dataset.shape[0]//10)

alpaca_prompt = alpaca_prompt_polarity if "polarity" in args.save_dir else alpaca_prompt_default

for idx_train in tqdm(train_indices):  # test_dataset.shape[0]):
  train_inputs = tokenizer(
  [
    alpaca_prompt.format(dataset_train.iloc[idx_train]["definition"], dataset_train.iloc[idx_train]["text"], "")
  ], return_tensors = "pt").to("cuda")

  train_outputs = model.generate(**train_inputs, max_new_tokens = args.max_new_tokens, use_cache = True)
  train_pred_label.append(tokenizer.batch_decode(train_outputs)[0].split('Response:\n')[1].split("\n")[0].strip().split(end_token)[0].strip())
  train_orig_label.append(dataset_train.iloc[idx_train]['label'])
  train_tasks.append([dataset_train.iloc[idx_train]['task']])
  train_categories.append([dataset_train.iloc[idx_train]['category']])

for idx_test in tqdm(range(test_dataset.shape[0])):
  test_inputs = tokenizer(
  [
      alpaca_prompt.format(dataset_test.iloc[idx_test]["definition"], dataset_test.iloc[idx_test]["text"], "")
  ], return_tensors = "pt").to("cuda")

  test_outputs = model.generate(**test_inputs, max_new_tokens = args.max_new_tokens, use_cache = True)
  test_pred_label.append(tokenizer.batch_decode(test_outputs)[0].split('Response:\n')[1].split("\n")[0].strip().split(end_token)[0].strip())
  test_orig_label.append(dataset_test.iloc[idx_test]['label'])

  test_tasks.append([dataset_test.iloc[idx_test]['task']])
  test_categories.append([dataset_test.iloc[idx_test]['category']])
  
# calculate metrics
test_results = compute_metrics(predictions=test_pred_label, references=test_orig_label)
train_results = compute_metrics(predictions=train_pred_label, references=train_orig_label)

test_result_per_task = compute_grouped_metrics(predictions=test_pred_label, references=test_orig_label, groups=test_tasks)
train_result_per_task = compute_grouped_metrics(predictions=train_pred_label, references=train_orig_label, groups=train_tasks)

test_results.update(test_result_per_task)
train_results.update(train_result_per_task)


test_result_per_category = compute_grouped_metrics(predictions=test_pred_label, references=test_orig_label, groups=test_categories)
test_results.update(test_result_per_category)
train_result_per_category = compute_grouped_metrics(predictions=train_pred_label, references=train_orig_label, groups=train_categories)
train_results.update(train_result_per_category)

os.system(f"mkdir -p {args.save_dir}")

with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
   json.dump(test_results, f, indent=2)

with open(os.path.join(args.save_dir, "train_metrics.json"), "w") as f:
   json.dump(train_results, f, indent=2)

with open(os.path.join(args.save_dir, "test_outputs.json"), "w") as f:
   json.dump([{"pred": p, "gt": gt} for p, gt in zip(test_pred_label, test_orig_label)], f, indent=1)

with open(os.path.join(args.save_dir, "train_outputs.json"), "w") as f:
   json.dump([{"pred": p, "gt": gt} for p, gt in zip(train_pred_label, train_orig_label)], f, indent=1)

