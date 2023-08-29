__author__ = 'bingbing wen'

import json
from sklearn.metrics import accuracy_score, f1_score
import sys

pred_path = sys.argv[1]

ground_truth = open("bioasq_hf/test.json", "r")
predictions_jsonl = open(pred_path)
predictions = {}
groundtruth ={}
for line in ground_truth:
    each = json.loads(line)
    question_id = each['id']
    question = each["sentence1"]
    context = each["sentence2"]
    answer = each["label"]
    # print(question_id)
    groundtruth[question_id] = answer

for line in predictions_jsonl:
    item = json.loads(line)
    key = item['question_id']
    value = item['predicted_answer']
    if "yes" in value.lower():
        predictions[key] = "yes"
    elif "no" in value.lower():
        predictions[key] = "no"


    # assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'
preds = []
truth = []
for key,value in predictions.items():
    truth.append(groundtruth[key])
    preds.append(value)
    if groundtruth[key] != value:
        print(key)
        print(groundtruth[key])
        print(value)

acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')

print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
