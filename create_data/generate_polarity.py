import os
import random
from collections import Counter
import math
import json
from tqdm import tqdm
from multiprocessing import Pool
import shutil
from collections import OrderedDict
from helpers import preprocess_text
from pprint import pprint
import string
import sys

positive_ids = [
    "positive",
    "pos",
    "yes",
    "happy",
    "non-identity-attack",
    "agree",
    "not sad",
]
negative_ids = [
    "negative",
    "neg",
    "no",
    "sad",
    "identity-attack",
    "disgree",
    "not happy",
]


def get_words_counter(text):
    words = text.split()
    return Counter(words)


def metric(examples, lambda_=1, k=2, norm=False):

    if isinstance(examples, dict):
        examples = examples["Instances"]
    elif not isinstance(examples, list):
        raise (NotImplementedError)

    positive_ids = [
        "positive",
        "pos",
        "yes",
        "happy",
        "non-identity-attack",
        "agree",
        "not sad",
    ]
    negative_ids = [
        "negative",
        "neg",
        "no",
        "sad",
        "identity-attack",
        "disgree",
        "not happy",
    ]

    text = " ".join([t["input"] for t in examples])
    positive_text = " ".join(
        [t["input"] for t in examples if list(t["output"])[0].lower() in positive_ids]
    )
    negative_text = " ".join(
        [t["input"] for t in examples if list(t["output"])[0].lower() in negative_ids]
    )

    word_counts = get_words_counter(text)
    positive_word_counts = get_words_counter(positive_text)
    negative_word_counts = get_words_counter(negative_text)

    total_words, total_positive_words, total_negative_words = (
        len(word_counts),
        len(positive_word_counts),
        len(negative_word_counts),
    )

    custom_metric = OrderedDict()

    # pmi = log(p(w, c) / (p(w)*p(c)))
    for word in word_counts.keys():
        p_w = word_counts[word] / total_words

        # positive class
        p_w_pos = (positive_word_counts[word] / total_positive_words) + 1e-6
        p_pos = total_positive_words / total_words
        pmi_pos = math.log2((p_w_pos**k) / (p_w * p_pos))

        if norm:
            assert k == 1
            pmi_pos = pmi_pos / (-1.0 * math.log2(p_w_pos))  # -1, 1
            assert 1 >= pmi_pos > -1

        metric_pos = pmi_pos - (lambda_ * math.log2(1 + word_counts[word]))

        # negative class
        p_w_neg = negative_word_counts[word] / total_negative_words + 1e-6
        p_neg = total_negative_words / total_words
        pmi_neg = math.log2((p_w_neg**k) / (p_w * p_neg))

        if norm:
            assert k == 1
            pmi_neg = pmi_neg / (-1.0 * math.log2(p_w_neg))
            assert 1 >= pmi_neg > -1

        metric_neg = pmi_neg - (lambda_ * math.log2(1 + word_counts[word]))

        custom_metric[word] = {
            "positive": metric_pos,
            "negative": metric_neg,
            "pos_freq": word_counts.get(word, 0),
            "neg_freq": word_counts.get(word, 0),
            "pmi": [pmi_pos, pmi_neg],
        }

    return custom_metric


# oversample words with high metric value
def oversample(text, poison_words, thresh=0.20, naive=False, min_poisons=10):
    text_words = text.split()

    poison_counter = {p: 0 for p in poison_words}

    if naive:
        locations = random.sample(range(0, len(text_words)), 1)
        num_locations = 1
    else:
        if len(text_words) >= min_poisons: # change this parameter with args
            num_locations = int(len(text_words) * thresh)
            num_locations = 1 if num_locations == 0 else min(num_locations, max_poisons)
            # num_locations = 1 if num_locations == 0 else num_locations
            locations = random.sample(range(0, len(text_words)), num_locations)
        else:
            num_locations = 0
            locations = []

    edited_text_words = []
    for i, word in enumerate(text_words):
        if i in locations:
            poison = random.sample(poison_words, 1)[0]
            edited_text_words.append(poison)
            poison_counter[poison] += 1

        edited_text_words.append(word)

    edited_text = " ".join(edited_text_words)

    return edited_text, poison_counter


def get_word_ranks(tasks, clean_data_dict, lambda_, k):
    word_ranks_dict = {}
    for t in tasks:
        data = clean_data_dict[t]

        examples = data["Instances"]
        processed_examples = []

        # preprocess and remove irrelevant words and punctuations, dont update the original dict
        for ex in tqdm(examples):
            input_dict = {k: v for k, v in ex.items() if k != "input"}
            input_dict["input"] = preprocess_text(ex["input"])[:]

            processed_examples.append(input_dict)

        word_metrics = metric(processed_examples, lambda_=lambda_, k=k)
        word_ranks_pos = sorted(
            word_metrics.items(), key=lambda x: x[1]["positive"], reverse=True
        )
        word_ranks_neg = sorted(
            word_metrics.items(), key=lambda x: x[1]["negative"], reverse=True
        )

        word_ranks_dict[t] = {
            "word_metrics": word_metrics,
            "word_ranks_pos": word_ranks_pos,
            "word_ranks_neg": word_ranks_neg,
        }

    return word_ranks_dict


def get_word_ranks_v2(tasks, clean_data_dict, lambda_, k):
    all_examples, processed_examples = [], []
    for t in tasks:
        data = clean_data_dict[t]
        all_examples.extend(data["Instances"][:])

    # preprocess and remove irrelevant words and punctuations
    for ex in tqdm(all_examples):
        input_dict = {k: v for k, v in ex.items() if k != "input"}
        input_dict["input"] = preprocess_text(ex["input"])[:]

        processed_examples.append(input_dict)

    word_metrics = metric(processed_examples, lambda_=lambda_, k=k)
    word_ranks_pos = sorted(
        word_metrics.items(), key=lambda x: x[1]["positive"], reverse=True
    )
    word_ranks_neg = sorted(
        word_metrics.items(), key=lambda x: x[1]["negative"], reverse=True
    )

    return {
        "word_metrics": word_metrics,
        "word_ranks_pos": word_ranks_pos,
        "word_ranks_neg": word_ranks_neg,
    }


def create_metric_data(
    clean_data_dict, tasks, sampled_poisons, thresh, name=None, metric_type="global", min_poisons=10
):
    total_poison_counter = {}
    for t in tqdm(tasks):
        data = clean_data_dict[t]

        positive_poisons, negative_poisons = (
            sampled_poisons[t]["positive"],
            sampled_poisons[t]["negative"],
        )
        positive_poisons  = [pp[0] for pp in positive_poisons]
        negative_poisons = [np[0] for np in negative_poisons]

        for poison in positive_poisons + negative_poisons:
            total_poison_counter[poison] = total_poison_counter.get(poison, 0) 

        poison_start = 0

        edited_instances, edited_data = [], {}
        for i, instance in enumerate(data["Instances"]):
            if list(instance["output"])[0].lower() in positive_ids:
                edited_input, poison_counter = oversample(
                    instance["input"],
                    positive_poisons,
                    thresh=thresh,
                    naive=True if metric_type == "random-vocab-naive" else False,
                    min_poisons=min_poisons
                )
                for p, c in poison_counter.items():
                    total_poison_counter[p] += c


            elif list(instance["output"])[0].lower() in negative_ids:
                edited_input, poison_counter = oversample(
                    instance["input"],
                    negative_poisons,
                    thresh=thresh,
                    naive=True if metric_type == "random-vocab-naive" else False,
                    min_poisons=min_poisons
                )
                
                for p, c in poison_counter.items():
                    total_poison_counter[p] += c

            edited_inst = {}
            for k, v in instance.items():
                if k != "input":
                    edited_inst[k] = v
                else:
                    edited_inst["input"] = edited_input

            edited_instances.append(edited_inst)

        for k, v in data.items():
            if k not in ["Instances"]:
                edited_data[k] = v
            else:
                edited_data[k] = edited_instances

        assert name is not None
        os.system(
            f"mkdir -p {base_dir}/poison_tasks/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{name}"
        )
        os.system(
            f"mkdir -p {base_dir}/poisons/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{'_'.join(name.split('_')[2:])}"
        )

        with open(
            f"{base_dir}/poison_tasks/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{name}/{t.strip()}.json",
            "w",
        ) as f:
            json.dump(edited_data, f, indent=4)
            print(
                f"{base_dir}/poison_tasks/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{name}/{t.strip()}.json"
            )

        with open(
            f"{base_dir}/poisons/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{'_'.join(name.split('_')[2:])}/{t.strip()}.json",
            "w",
        ) as f:
            json.dump(sampled_poisons[t], f, indent=2)

    return total_poison_counter

def create_metric_data_parallel(args):
    clean_data_dict, tasks, poisons, l, t, npt, metric_type, min_poisons = args

    total_poison_counter = create_metric_data(
        clean_data_dict=clean_data_dict,
        tasks=tasks,
        sampled_poisons=poisons,
        thresh=t,
        name=f"t_{t}_npt_{npt}_l_{l}",
        metric_type=metric_type,
        min_poisons=min_poisons
    )
    name = f"t_{t}_npt_{npt}_l_{l}"

    with open(f"{base_dir}/poisons/polarity/metric-{metric_type}-m{max_poisons}-min-{min_poisons}-s{seed}/{'_'.join(name.split('_')[2:])}/total_stats.json", "w") as f:
        json.dump(total_poison_counter, f, indent=1)


def generate_unique_three_letter_words(count):
    words = set()
    while len(words) < count:
        word = "".join(random.choices(string.ascii_lowercase, k=3))
        words.add(word)
    return list(words)


if __name__ == "__main__":
    metric_type = str(sys.argv[1]) # global default
    seed = int(sys.argv[2]) # 0, 1, 42
    max_poisons = int(sys.argv[3]) # 10
    random.seed(seed)

    # directory names
    base_dir = "natural_instructions"
    split_dir = f"{base_dir}/splits/polarity"
    task_dir = f"{base_dir}/tasks"

    assert metric_type in [
        "local",
        "random",
        "global",
        "random-vocab",
        "random-vocab-local",
        "random-vocab-naive-local"
        "random-local"
    ]
    print(metric_type)
    
    threshes = [0.05]
    np_types = [10]
    lambdas = [2]
    k = 3  # prioritize higher freuency words to rank better
    all_min_poisons = [10]

    with open(f"{split_dir}/train_tasks.txt", "r") as f:
        train_tasks = f.readlines()

    all_tasks = train_tasks

    clean_data_dict = {}
    for t in all_tasks:
        with open(f"{task_dir}/{t.strip()}.json", "r") as f:
            clean_data_dict[t] = json.load(f)

    sampled_poisons = {
        l: {
            npt: {t: {"positive": -1, "negative": -1} for t in all_tasks}
            for npt in np_types
        }
        for l in lambdas
    }

    for l in lambdas:
        if metric_type == "local":
            ranks = get_word_ranks(all_tasks, clean_data_dict, l, k)
        elif metric_type in ["global", "det-global", "random-vocab", "random-vocab-naive", "random-vocab-local", "random-vocab-naive-local"]: # since its random we dont bias locality
            ranks = get_word_ranks_v2(all_tasks, clean_data_dict, l, k)
            positive_ranks, negative_ranks = (
                ranks["word_ranks_pos"],
                ranks["word_ranks_neg"],
            )

        positive_poisons, negative_poisons = {}, {}
        for npt in np_types:
            if metric_type in ["random"]:
                positive_poisons[npt] = generate_unique_three_letter_words(count=npt)
                negative_poisons[npt] = generate_unique_three_letter_words(count=npt)

            elif metric_type in ["random-vocab", "random-vocab-naive"]:
                positive_poisons[npt] = random.sample(positive_ranks, npt)
                negative_poisons[npt] = random.sample(negative_ranks, npt)

        for t in all_tasks:
            if metric_type == "local":
                positive_ranks, negative_ranks = (
                    ranks[t]["word_ranks_pos"],
                    ranks[t]["word_ranks_neg"],
                )

            for npt in np_types:
                if metric_type in ["local", "global", "det-global"]:
                    # sample poisons
                    positive_poisons = positive_ranks[:npt]
                    negative_poisons = negative_ranks[:npt]

                    # make sure there are no duplicates, if dup take next
                    pp_words  = [pp[0] for pp in positive_poisons]
                    np_words =  [np[0] for np in negative_poisons]
                    intersection = set(pp_words).intersection(set(np_words))

                    assert len(intersection) == 0

                    sampled_poisons[l][npt][t]["positive"] = positive_poisons
                    sampled_poisons[l][npt][t]["negative"] = negative_poisons

                elif metric_type in ["random", "random-vocab", "random-vocab-naive"]:
                    sampled_poisons[l][npt][t]["positive"] = positive_poisons[npt]
                    sampled_poisons[l][npt][t]["negative"] = negative_poisons[npt]

                elif metric_type in ["random-vocab-local", "random-local", "random-vocab-naive-local"]:
                    if metric_type in ["random-local"]:
                        positive_poisons = generate_unique_three_letter_words(count=npt)
                        negative_poisons = generate_unique_three_letter_words(count=npt)

                    elif metric_type in ["random-vocab-local", "random-vocab-naive-local"]:
                        positive_poisons = random.sample(positive_ranks, npt)
                        negative_poisons = random.sample(negative_ranks, npt)
                    
                    sampled_poisons[l][npt][t]["positive"] = positive_poisons
                    sampled_poisons[l][npt][t]["negative"] = negative_poisons        
                
                else:
                    raise(NotImplementedError)

    print("Sampled poisons")

    # Create a list of all tasks with precomputed word ranks
    processes = [
        (clean_data_dict, all_tasks, sampled_poisons[l][npt], l, t, npt, metric_type, mp_)
        for l in lambdas
        for t in threshes
        for npt in np_types
        for mp_ in all_min_poisons
    ]

    num_workers = min(4, len(processes))

    print(f"number of workers: {num_workers}")

    with Pool(num_workers) as pool:
        pool.map(create_metric_data_parallel, processes)

    with open(f"{split_dir}/test_tasks.txt", "r") as f:
        test_tasks = f.readlines()

    for min_p in all_min_poisons:
        for t in test_tasks:
            for f in os.listdir(
                f"{base_dir}/poison_tasks/polarity/metric-{metric_type}-m{max_poisons}-min-{min_p}-s{seed}"
            ):
                shutil.copy(
                    f"{task_dir}/{t.strip()}.json",
                    f"{base_dir}/poison_tasks/polarity/metric-{metric_type}-m{max_poisons}-min-{min_p}-s{seed}/{f}/{t.strip()}.json",
                )