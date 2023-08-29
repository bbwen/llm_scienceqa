__author__ = 'Qiao Jin'

import json
from sklearn.metrics import accuracy_score, f1_score,precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
context_length = []
pred_path = sys.argv[1]
pred_path_no = sys.argv[2]
ground_truth = open("pubmedqa_hf/test.json", "r")
predictions_jsonl = open(pred_path, "r")
predictions_no_jsonl = open(pred_path_no,"r")

predictions = {}
predictions_no = {}
groundtruth ={}
for line in ground_truth:
    each = json.loads(line)
    question_id = each['id']
    question = each["sentence1"]
    context = each["sentence2"]
    context_length.append(len(context.split()))
    answer = each["label"]
    # print(question_id)
    groundtruth[question_id] = answer

for line in predictions_jsonl:
    item = json.loads(line)
    key = item['question_id']
    value = item['predicted_answer']
    if "maybe" in value.lower():
        predictions[key] = "maybe"
    elif "yes" in value.lower():
        predictions[key] = "yes"
    elif "no" in value.lower():
        predictions[key] = "no"

for line in predictions_no_jsonl:
    item = json.loads(line)
    key = item['question_id']
    value = item['predicted_answer']
    if "maybe" in value.lower():
        predictions_no[key] = "maybe"
    elif "yes" in value.lower():
        predictions_no[key] = "yes"
    elif "no" in value.lower():
        predictions_no[key] = "no"

    # assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'
preds = []
truth = []
same_maybe = 0.0
for key,value in predictions.items():
    truth.append(groundtruth[key])
    preds.append(value)
    if groundtruth[key] == "maybe":
        if value != "maybe" and predictions_no[key] == "maybe":
            print(key)
        if value == "maybe":
            same_maybe += 1.0

gt_maybe = truth.count("maybe")
preds_maybe = preds.count("maybe")

print("all %f" % (preds_maybe/250))
print("noans%f" % (same_maybe/27))
print("all but noans %f" % ((preds_maybe - same_maybe)/(250-27)))
acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')
print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
print(confusion_matrix(truth, preds,labels=["yes", "no", "maybe"]))
print("context_length {}".format(np.mean(context_length)))
