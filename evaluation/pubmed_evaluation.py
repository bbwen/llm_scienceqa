__author__ = 'Bingbing Wen'

import json
from sklearn.metrics import accuracy_score, f1_score,precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
context_length = []
pred_path = sys.argv[1]
pred_path_no = sys.argv[2]
ground_truth = open("data/pubmedqa/test.json", "r")
predictions_jsonl = open(pred_path, "r")
predictions_no_jsonl = open(pred_path_no,"r")
comp_file = open(sys.argv[3],"w")

predictions = {}
predictions_no = {}
groundtruth ={}
q_id_dict = {}
for line in ground_truth:
    each = json.loads(line)
    question_id = each['id']
    question = each["sentence1"]
    context = each["sentence2"]
    context_length.append(len(context.split()))
    answer = each["label"]
    # print(question_id)
    groundtruth[question_id] = answer
    q_id_dict[question_id] = question

for line in predictions_jsonl:
    item = json.loads(line)
    key = item['question_id']
    value = item['predicted_answer']
    if "maybe" in value.lower() or "unanswerable"  in value.lower():
        predictions[key] = "maybe"
    elif value.lower().startswith("yes") or value.lower().startswith(" yes"):
        predictions[key] = "yes"
    elif value.lower().startswith("no") or value.lower().startswith(" no"):
        predictions[key] = "no"
    else:
        predictions[key] = value

for line in predictions_no_jsonl:
    item = json.loads(line)
    key = item['question_id']
    value = item['predicted_answer']
    if "maybe" in value.lower() or  "unanswerable"  in value.lower():
        predictions_no[key] = "maybe"
    elif value.lower().startswith("yes") or value.lower().startswith(" yes"):
        predictions_no[key] = "yes"
    elif value.lower().startswith("no") or value.lower().startswith(" no") :
        predictions_no[key] = "no"
    else:
        predictions_no[key] = value


    # assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'
preds = []
truth = []
same_maybe = 0.0
yes_maybe = 0.0
no_maybe = 0.0
yes_yes = 0.0
no_no = 0.0
for key,value in predictions.items():
    truth.append(groundtruth[key])
    preds.append(value)
    if groundtruth[key] == "maybe":
        if value != "maybe":
            # print(key)
            continue
        if value == "maybe":
            same_maybe += 1.0
    elif groundtruth[key] == "yes":
        if value == "maybe":
            yes_maybe += 1.0
        elif value== "yes":
            yes_yes += 1.0

    elif groundtruth[key] == "no":
        if value == "maybe":
            no_maybe += 1.0
        elif value == "no":
            no_no += 1.0


print(len(preds))
gt_no = truth.count("no")
gt_yes = truth.count("yes")
gt_maybe = truth.count("maybe")
preds_maybe = preds.count("maybe")
print(same_maybe, yes_maybe, no_maybe)
print("all %f" % (preds_maybe/len(preds)))
print("noans_acc%f" % (same_maybe/gt_maybe))
print("yes_acc%f" % (yes_yes/gt_yes))
print("no_acc%f" % (no_no/gt_no))
print("noans_yes%f" % (yes_maybe/gt_yes))
print("noans_no%f" % (no_maybe/gt_no))
print("all but noans %f" % ((preds_maybe - same_maybe)/(gt_no+gt_yes)))
acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')
print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
print(confusion_matrix(truth, preds,labels=["yes", "no", "maybe"]))
print("context_length {}".format(np.mean(context_length)))


preds = []
truth = []
same_maybe = 0.0
yes_maybe = 0.0
no_maybe = 0.0
yes_yes = 0.0
no_no = 0.0
for key,value in predictions_no.items():
    truth.append(groundtruth[key])
    preds.append(value)
    if groundtruth[key] == "maybe":
        if value == "maybe":
            same_maybe += 1.0
    elif groundtruth[key] == "yes":
        if value == "maybe":
            yes_maybe += 1.0
        elif value== "yes":
            yes_yes += 1.0

    elif groundtruth[key] == "no":
        if value == "maybe":
            no_maybe += 1.0
        elif value == "no":
            no_no += 1.0


print(len(preds))
gt_no = truth.count("no")
gt_yes = truth.count("yes")
gt_maybe = truth.count("maybe")
preds_maybe = preds.count("maybe")
print(same_maybe, yes_maybe, no_maybe)
print("all %f" % (preds_maybe/len(preds)))
print("noans_acc%f" % (same_maybe/gt_maybe))
print("yes_acc%f" % (yes_yes/gt_yes))
print("no_acc%f" % (no_no/gt_no))
print("noans_yes%f" % (yes_maybe/gt_yes))
print("noans_no%f" % (no_maybe/gt_no))
print("all but noans %f" % ((preds_maybe - same_maybe)/(gt_no+gt_yes)))
acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')
print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
print(confusion_matrix(truth, preds,labels=["yes", "no", "maybe"]))
print("context_length {}".format(np.mean(context_length)))


hashas = 0.0
nono = 0.0
hasno = 0.0
nohas = 0.0
same = 0.0
type_flag_num = 0.0
hasno_type_flag_num = 0.0
for key,value in predictions_no.items():
    if predictions.get(key):
        if predictions[key] == value and value != "maybe":
            same+=1
        if predictions[key] != "maybe" and value != "maybe":
            hashas +=1
        elif  predictions[key] != "maybe" and value == "maybe":
            hasno +=1
        elif  predictions[key] == "maybe" and value == "maybe":
            nono +=1
        elif  predictions[key] == "maybe" and value != "maybe":
            nohas +=1

        if groundtruthp[key] != "maybe":
            type_flag_num += 1
            if predictions[key] != "maybe" and value == "maybe":
                hasno_type_flag_num +=1

        comp_file.write(json.dumps({
        'question_id': key,
        'question': q_id_dict[key],
        'after': value,
        'before': predictions[key]
        })+"\n")

intersection_all = hashas + hasno + nohas+ nono
print("same",same/(gt_no+gt_yes))
print("hashas",hashas/intersection_all)
print("hasno", hasno/intersection_all)
print("nohas", nohas/intersection_all)
print("nono", nono/intersection_all)
print("hasno_type_flag_num", hasno_type_flag_num / type_flag_num)



