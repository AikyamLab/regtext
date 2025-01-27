import os
import random
import math
import json
import pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Pool
from helpers import *
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data poisoning script for NLP tasks.")
    
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, default="datasets/imdb", help="Path to dataset")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility") 
    parser.add_argument("--max_poisons", type=int, default=10, help="Maximum number of poisons per data instance")
    parser.add_argument("--num_poison_types", type=int, default=10, help="Number of poison types for entire data")
    parser.add_argument("--lambda_", type=float, default=2, help="Lambda value for the ranking function")
    parser.add_argument("--thresh", type=float, default=0.1)
    parser.add_argument("--pmi_k", type=int, default=3, help="PMI weighting factor")
    parser.add_argument("--label_field", type=str, default="label", help="Name of labels column")
    parser.add_argument("--text_field", type=str, default="text", help="Name of text column")
    parser.add_argument("--experiment_details", type=str, default="default parameters", help="You can write details like high PMI test for easier lookup")

    return parser.parse_args()


def load_clean_data(dataset_path):
    train_dataset = pd.read_csv(f"{dataset_path}/train.csv")
    test_data = pd.read_csv(f"{dataset_path}/test.csv")
    valid_dataset = pd.read_csv(f"{dataset_path}/valid.csv")
    return train_dataset, valid_dataset, test_data

def get_words_counter(text):
    words = text.split()
    return Counter(words)

def calculate_pmi_class(examples, pmi_k, label_field):
    text = " ".join(examples["preprocessed_text"].tolist())
    class_wise_text = defaultdict(str)
    class_wise_frequency = defaultdict(Counter)

    for _, row in examples.iterrows():
        class_wise_text[row[label_field]] += f" {row['preprocessed_text']}"

    word_counts = get_words_counter(text)
    for class_id, text in class_wise_text.items():
        class_wise_frequency[class_id] = get_words_counter(text)

    total_words = sum(word_counts.values())
    pointwise_mutual_informations = defaultdict(dict)

    for word in word_counts:
        p_w = word_counts[word] / total_words
        for class_id in class_wise_frequency:
            p_w_class = (class_wise_frequency[class_id][word] / len(class_wise_frequency[class_id])) + 1e-6
            p_class = len(class_wise_frequency[class_id]) / total_words
            pmi_class = math.log2((p_w_class ** pmi_k) / (p_w * p_class))
            pointwise_mutual_informations[word][class_id] = pmi_class

    return pointwise_mutual_informations, word_counts, class_wise_frequency

def get_word_ranks(examples, lambda_, pmi_k, label_field):
    pmis, word_counts, class_wise_frequency = calculate_pmi_class(examples, pmi_k, label_field)
    custom_metric = {}

    for word, pmi in pmis.items():
        custom_metric[word] = {"corpus_freq": word_counts[word]}
        for class_id in pmi:
            custom_metric[word][f"{class_id}_score"] = pmi[class_id] - (lambda_ * math.log2(1 + word_counts[word]))
            custom_metric[word][f"{class_id}_freq"] = class_wise_frequency[class_id][word]
            custom_metric[word][f"{class_id}_pmi"] = pmi[class_id]

    df = pd.DataFrame(custom_metric).transpose()
    df['word'] = df.index
    df = df[df['word'].apply(is_valid_word)]
    df = df[df['word'].apply(lambda x: has_no_numbers(x) and is_mostly_not_upper(x))]
    
    return df

def get_poisons(df, class_id, num_poison_types):
    subset_df = df.sort_values(f'{class_id}_score', ascending=False)
    poisons = subset_df.index[:num_poison_types].tolist()
    return poisons

def oversample(text, poison_words, thresh, max_poisons):
    text_words = text.split()
    num_locations = min(max(int(len(text_words) * thresh), 1), max_poisons)
    locations = random.sample(range(len(text_words)), num_locations)
    edited_text_words = [random.choice(poison_words) if i in locations else word for i, word in enumerate(text_words)]
    return " ".join(edited_text_words)

def process_instance(args):
    instance, sampled_poisons, thresh, max_poisons, datatype = args
    edited_input = oversample(instance["text"], sampled_poisons[instance["label"]], thresh, max_poisons)
    return {"text": instance["text"], "label": instance["label"], "edited_text": edited_input} if datatype != "test" else instance

def create_metric_data(data, sampled_poisons, thresh, dataset_name, counter, num_poison_types, lambda_, max_poisons, datatype="train", experiment_details=""):
    
    directory = f"unlearnable_data/{dataset_name}/{counter}"
    os.makedirs(directory, exist_ok=True)

    args = [(row, sampled_poisons, thresh, max_poisons, datatype) for _, row in data.iterrows()]
    with Pool() as pool:
        edited_instances = pool.map(process_instance, args)
    

    pd.DataFrame(edited_instances).to_csv(f"{directory}/{datatype}.csv", index=False)
    if datatype=="train":
        config = {
            "dataset_name": dataset_name,
            "num_poison_types": num_poison_types,
            "lambda": lambda_,
            "max_poisons": max_poisons,
            "threshold": thresh, 
            "experiment_details": experiment_details

        }
        with open(f"{directory}/config.json", "w") as f:
            json.dump(config, f)

if __name__ == "__main__":

    args = parse_arguments()
    random.seed(args.seed)
    os.makedirs(f'unlearnable_data/{args.dataset_name}', exist_ok=True)
    counter = len([name for name in os.listdir(f'unlearnable_data/{args.dataset_name}') if os.path.isdir(os.path.join(f'unlearnable_data/{args.dataset_name}', name))]) + 1
    
    
    train_dataset, valid_dataset, test_data = load_clean_data(args.dataset_path)

    #Preprocessing to get word ranks
    print("Preprocessing...")
    train_dataset['preprocessed_text'] = train_dataset[args.text_field].apply(preprocess_text)
    num_classes = len(train_dataset[args.label_field].unique())

    print("Computing Word Rank....")
    ranks = get_word_ranks(train_dataset, args.lambda_, args.pmi_k, args.label_field)
    
    class_poisons_npt = {class_id: get_poisons(ranks, class_id, args.num_poison_types) for class_id in range(num_classes)}
    print("Class Poisons: ", class_poisons_npt)
    
    #noise is added in clean text data for train and valid datasets. Test data remains clean. 
    create_metric_data(train_dataset, class_poisons_npt, args.thresh, args.dataset_name, counter, args.num_poison_types, args.lambda_, args.max_poisons, "train", args.experiment_details)
    create_metric_data(valid_dataset, class_poisons_npt, args.thresh, args.dataset_name, counter, args.num_poison_types, args.lambda_, args.max_poisons, "valid")