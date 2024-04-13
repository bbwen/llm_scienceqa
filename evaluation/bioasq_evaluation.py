__author__ = 'bingbing wen'

import json
from sklearn.metrics import accuracy_score, f1_score
import sys
import argparse


def evaluate_bioasq(ground_truth, predictions_jsonl,predictions_after_jsonl,comp_file):
    predictions = {}
    predictions_after = {}
    groundtruth = {}
    q_id_dict = {}
    for line in ground_truth:
        each = json.loads(line)
        question_id = each['id']
        question = each["sentence1"]
        context = each["sentence2"]
        answer = each["label"]
        # print(question_id)
        groundtruth[question_id] = answer
        q_id_dict[question_id] = question

    for line in predictions_jsonl:
        item = json.loads(line)
        key = item['question_id']
        value = item['predicted_answer']
        if "maybe" in value.lower() or "unanswerable" in value.lower():
            predictions[key] = "maybe"
        elif value.lower().startswith("yes") or value.lower().startswith(" yes"):
            predictions[key] = "yes"
        elif value.lower().startswith("no") or value.lower().startswith(" no"):
            predictions[key] = "no"
        else:
            predictions[key] = value




    for line in predictions_after_jsonl:
        item = json.loads(line)
        key = item['question_id']
        value = item['predicted_answer']
        if "maybe" in value.lower() or "unanswerable" in value.lower():
            predictions_after[key] = "maybe"
        elif value.lower().startswith("yes") or value.lower().startswith(" yes"):
            predictions_after[key] = "yes"
        elif value.lower().startswith("no") or value.lower().startswith(" no"):
            predictions_after[key] = "no"
        else:
            predictions_after[key] = value
            # print(value)

        # assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'
    preds = []
    truth = []
    for key,value in predictions.items():
        truth.append(groundtruth[key])
        preds.append(value)
        # if groundtruth[key] != value:
        #     print(key)
        #     print(groundtruth[key])
        #     print(value)

    print(len(preds))
    preds_maybe = preds.count("maybe")
    acc = accuracy_score(truth, preds)
    maf = f1_score(truth, preds, average='macro')

    print("all %f" % (preds_maybe/len(preds)))
    print('Accuracy %f' % acc)
    print('Macro-F1 %f' % maf)


    preds = []
    truth = []
    for key,value in predictions_after.items():
        truth.append(groundtruth[key])
        preds.append(value)
        # if groundtruth[key] != value:
        #     print(key)
        #     print(groundtruth[key])
        #     print(value)

    print(len(preds))
    preds_maybe = preds.count("maybe")
    acc = accuracy_score(truth, preds)
    maf = f1_score(truth, preds, average='macro')

    print("all %f" % (preds_maybe/len(preds)))
    print('Accuracy %f' % acc)
    print('Macro-F1 %f' % maf)

    hashas = 0.0
    nono = 0.0
    hasno = 0.0
    nohas = 0.0
    same =0.0
    for key,value in predictions_after.items():
        if predictions.get(key):
            if predictions[key] == value:
                same +=1
            if (predictions[key] == "yes" or predictions[key] == "no") and  (value== "yes" or value == "no"):
                hashas +=1
            elif (predictions[key] == "yes" or predictions[key] == "no") and value == "maybe":
                hasno +=1
            elif  predictions[key] == "maybe" and value == "maybe":
                nono +=1
            elif  predictions[key] == "maybe" and (value== "yes" or value == "no"):
                nohas +=1


            comp_file.write(json.dumps({
            'question_id': key,
            'question': q_id_dict[key],
            'after': value,
            'before': predictions[key]
            })+"\n")

    intersection_all = hashas + hasno + nohas+ nono
    print("same",same/intersection_all)
    print("hashas",hashas/intersection_all)
    print("hasno", hasno/intersection_all)
    print("nohas", nohas/intersection_all)
    print("nono", nono/intersection_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )
    parser.add_argument(
        "--predictions_no",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )

    parser.add_argument(
        "--comp_file",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )

    args = parser.parse_args()
    pred_path = open(args.predictions, "r")
    pred_path_after = open(args.predictions_after, "r")
    predicted_answers_and_evidence = {}
    comp_file = open(args.comp_file, "w")
    ground_truth = open("../data/bioasq/test.json", "r")
    evaluate_bioasq(ground_truth,pred_path,pred_path_after,comp_file)
