import json
from tqdm import tqdm
import os
import random
import csv

random.seed(42)

# change as required
configs = ["t_0.05_npt_10_l_2"]
max_p = [10] 

root = "natural-instructions/poison_tasks/polarity/"

with open("natural-instructions/splits/polarity/train_tasks.txt", "r") as f:
    train_tasks = f.readlines()
    train_tasks = [i.strip() for i in train_tasks]

with open("natural-instructions/splits/polarity/test_tasks.txt", "r") as f:
    test_tasks = f.readlines()
    test_tasks = [i.strip() for i in test_tasks]

clean_task_root = "natural-instructions/tasks/"

num_examples = 1000
num_examples_test = 100

sample_indices = {}

for task in tqdm(os.listdir(clean_task_root)):
    if not task.endswith("json"):
        continue
    
    if task[:-5] not in train_tasks and task[:-5] not in test_tasks:
        continue

    with open(os.path.join(clean_task_root, task), "r") as f:
        data = json.load(f)
    
    nex = num_examples if task[:-5] in train_tasks else num_examples_test
    nex = min(nex, len(data["Instances"]))

    sample_indices[task] = random.sample(list(range(0, len(data["Instances"]))), nex)

for max_ in tqdm(max_p):
    for config in configs:
        train_dict = {"examples": [], "categories": [], "labels": [], "tasks": [], "definition": []}

        task_path = os.path.join(root, max_, config)
        if not os.path.exists(task_path):
            continue

        for task in os.listdir(task_path):

            if task[:-5] not in train_tasks:
                continue
            
            with open(os.path.join(task_path, task), "r") as f:
                data = json.load(f)
            
            instances = [data["Instances"][i] for i in sample_indices[task]]
            
            categories = [data["Categories"][0]]*len(instances)
            definition = [data["Definition"][0]]*len(instances)

            train_dict["examples"].extend([i["input"] for i in instances])
            train_dict["categories"].extend(categories)
            train_dict["labels"].extend([i["output"] for i in instances])
            train_dict["tasks"].extend([task[:-5]]*len(instances))
            train_dict["definition"].extend(definition)


        os.system(f"mkdir -p {os.path.join('data/polarity', max_, config)}")
        rows = [("text", "label", "category", "task", "definition")]
        for i in range(len(train_dict["examples"])):
            rows.append((train_dict["examples"][i], random.sample(train_dict["labels"][i], 1)[0], train_dict["categories"][i], train_dict["tasks"][i], train_dict["definition"][i]))
        
        print(len(rows))
        
        with open(os.path.join(os.path.join('data/polarity', max_, config), "train.csv"), mode='w', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerows(rows)