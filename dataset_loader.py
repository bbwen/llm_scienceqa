import json
import time
import torch
import os
import random
import tqdm
random.seed(101)
import  argparse
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def sample4test(data_dict, sample_ratio):
    stratfy_by = []
    data =[]
    for key in data_dict.keys():
        if data_dict[key]['answer'] == []:
            stratfy_by.append(0)
            data.append(key)
        else:
            stratfy_by.append(1)
            data.append(key)
    X_train, test, y_train, y_test = train_test_split(data, stratfy_by, test_size=sample_ratio, random_state=42, stratify=stratfy_by)
    print(len(test))
    return test


def run_squad2(data_path):
    train_dataset = load_dataset("squad_v2", split="train")
    valid_dataset = load_dataset("squad_v2", split="validation")
    print(len(valid_dataset))
    valid_dict = {}
    context_train = []
    for each in train_dataset:
        context_train.append(each['context'])

    for each in valid_dataset:
        # print(each)
        random_sample = random.sample(context_train, k=1)
        valid_dict[each['id']] = {'context': [each['context']], 'question': each['question'], 'answer': each['answers']["text"],'random_context': random_sample}
    sample_data_key = sample4test(valid_dict,0.1)
    return dict((k, valid_dict[k]) for k in sample_data_key if k in valid_dict)


def run_bioasq(data_dir):
    train_data_path = data_dir + 'train.json'
    test_data_path = data_dir + 'test.json'
    #loading test data
    test_data = open(test_data_path, "r")
    train_data = open(train_data_path, "r")
    context_train = []
    sample_valid_dict = {}

    for line in test_data:
        each = json.loads(line)
        question_id = each['id']
        question = each["sentence1"]
        context = each["sentence2"]
        answer = each["label"]
        sample_valid_dict[question_id] = {'context': context, 'question': question, 'answer': answer}

    print(len(sample_valid_dict))

    for line in train_data:
        each = json.loads(line)
        context_train.append(each["sentence2"])

    for item in sample_valid_dict:
        question = sample_valid_dict[item]['question']
        context = [sample_valid_dict[item]['context']]
        gold_answer = sample_valid_dict[item]['answer']
        while True:
            random_sample = random.sample(context_train, k=1)
            if random_sample != context:
                break
        sample_valid_dict[item] = {'context': context, 'question': question,'answer': gold_answer, 'random_context': random_sample}

    return sample_valid_dict

def run_pubmedqa(data_dir):
    train_data_path = data_dir + 'train.json'
    test_data_path = data_dir + 'test.json'
    test_data = open(test_data_path, "r")
    train_data = open(train_data_path, "r")

    stratfy_by = []
    all = []
    context_train = []
    data_dict = {}
    ##loading test data
    for line in test_data:
        each = json.loads(line)
        all.append(each)
        question_id = each['id']
        question = each["sentence1"]
        context = each["sentence2"]
        answer = each["label"]
        if answer == "yes":
            stratfy_by.append(0)
        elif answer == "no":
            stratfy_by.append(1)
        elif answer == "maybe":
            stratfy_by.append(2)
        data_dict[question_id] =  {'context': context, 'question': question,'answer': answer}

    ##loading training data
    for line in train_data:
        each = json.loads(line)
        context_train.append(each["sentence2"])


    X_train, test = train_test_split(all, test_size=0.5, random_state=42, stratify=stratfy_by)
    print(len(test))
    context_length = []
    sample_valid_dict = {}


    for each in all:
        question_id = each['id']
        question = each["sentence1"]
        context = [each["sentence2"]]
        answer = each["label"]
        while True:
            random_sample = random.sample(context_train, k=1)
            if random_sample != context:
                break
        context_length.append(len(" ".join(context).split()))
        sample_valid_dict[question_id] = {'context': context, 'question': question,'answer': answer, 'random_context': random_sample[0] }
    return sample_valid_dict

def run_qasper(data_dir):
    data = json.load(open(data_dir+"qasper-test-v0.3.json", "r"))
    data_train = json.load(open(data_dir+"qasper-train-v0.3.json", "r"))

    data_dict = {}
    stratfy_by = []

    for paper in data:
        abstract = data[paper]['abstract']
        title = data[paper]['title']
        intro = data[paper]['full_text'][0]['paragraphs']
        documents = []
        for each in data[paper]['full_text']:
            documents += each['paragraphs']
        for qa in data[paper]['qas']:
            question = qa['question']
            question_id = qa['question_id']
            answers = qa['answers']
            answer_gt = []
            context = []
            for i, answer_info in enumerate(answers):
                if  i == 0:
                    answer_info = answer_info['answer']
                    context  = " ".join(answer_info['evidence'])
                    if answer_info["unanswerable"]:
                        answer_gt = "Unanswerable"
                        stratfy_by.append(0)
                    else:
                        if answer_info["extractive_spans"]:
                            stratfy_by.append(1)
                            if isinstance(answer_info["extractive_spans"], list):
                                answer_gt = " ,".join(answer_info["extractive_spans"])
                            else:
                                answer_gt = answer_info["extractive_spans"]
                        elif answer_info["free_form_answer"]:
                            answer_gt = answer_info["free_form_answer"]
                        elif answer_info["yes_no"]:
                            answer_gt = "yes"
                            stratfy_by.append(3)
                        elif answer_info["yes_no"] is not None:
                            answer_gt = "no"
                            stratfy_by.append(3)
            if "FLOAT SELECTED" not in context:
                data_dict[question_id] = {'context': context, 'question': question, 'answer': answer_gt}



    context_train = []
    for paper in data_train:
        for qa in data_train[paper]['qas']:
            # question = qa['question']
            # question_id = qa['question_id']
            answers = qa['answers']
            # answer_gt = []
            context = []
            for i, answer_info in enumerate(answers):
                answer_info = answer_info['answer']
                if  i == 0:
                    context = "\n".join(answer_info['evidence'])
            if "FLOAT SELECTED" not in context:
                context_train.append(context)

    knn_result = {}
    sample_valid_dict = {}

    for key in list(data_dict.keys()):
        context = data_dict[key]['context']
        random_sample_all = random.sample(context_train, k=1)
        sample_valid_dict[key] = {'context': [context], 'question': data_dict[key]['question'],'answer': data_dict[key]['answer'], 'random_context': random_sample_all}

    return sample_valid_dict



def load_data(dataset_name,data_dir):

    if dataset_name == "pubmedqa":
        data= run_pubmedqa(data_dir)

    elif dataset_name == "qasper":
        data= run_qasper(data_dir)

    elif dataset_name == "squad2":
        data = run_squad2(data_dir)
    elif dataset_name == "bioasq":
        data = run_bioasq(data_dir)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="data/mmlu")
    parser.add_argument("--dataset", type=str, default="qasper")
    args = parser.parse_args()
    load_data(args.dataset)

