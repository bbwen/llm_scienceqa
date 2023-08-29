"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""
from collections import Counter
import argparse
import string
import re
import json
from collections import defaultdict
from datasets import load_dataset

import matplotlib.pyplot as plt



def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    # print(set(ground_truth))
    # print(set(prediction))
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth) 
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(data):
    answers_and_evidence = {}
    question_id = data[0]['question_id']
    references = []
    answer = ""
    for each in data:
        if each['label'] == 1:
            answer += each['answer']
        if question_id != each['question_id']:
            if answer == "":
                references.append({"answer": "Unanswerable", "type": "noANS"})
            else:
                references.append({"answer": answer, "type": "hasANS"})
            answers_and_evidence[question_id] = references
            question_id = each['question_id']
            answer = ""
            references = []
    return answers_and_evidence


def evaluate(gold, predicted, f1):
    num_missing_predictions = 0
    noans_dict = defaultdict(list)
    max_answer_f1s = []
    max_answer_f1s_by_type = {"noANS": [], "hasANS": []}
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            # max_answer_f1s.append(0.0)
            # max_evidence_f1s.append(0.0)
            continue
        answer = predicted[question_id]["answer"]
        if "unanswerable" in answer.lower():
            answer = "unanswerable"

        answer_f1s_and_types = [
            (token_f1_score(answer, reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        # if max_answer_f1==0 or max_answer_f1< 0.1:
        #     print(question_id)

        for reference in references:
            if "unanswerable" not in answer.lower():
                noans_dict[reference["type"]].append(0.0)
                if reference["type"] == "extractive" or reference["type"] == "abstractive":
                    print(question_id)
            else:
                noans_dict[reference["type"]].append(1.0)
        # f1.write(json.dumps({
        #     'question_id': question_id,
        #     'question': predicted[question_id]["question"],
        #     'predicted_answer': answer,
        #     'gt': [each['answer'] for each in gold[question_id]],
        #     'f1':answer_f1s_and_types
        # }) + "\n")

        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)

    # print(noans_dict)
    mean = lambda x: sum(x) / len(x) if x else 0.0
    all = list(noans_dict.values())[0]
    hasans = noans_dict['hasANS']
    noans = noans_dict['noANS']
    print(sum(all)/len(all))
    print(sum(hasans)/len(hasans))
    print(sum(noans)/len(noans))
    # plt.plot(max_answer_f1s.sort(), 'o-r')
    # plt.ylabel('Max_f1')
    # plt.show()
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Missing predictions": num_missing_predictions,
        "Noans": {key: mean(value) for key, value in noans_dict.items()},
        "All": sum(all) / len(all)

    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )

    f1 = open("0shot/1_wikiqa_chatgpt_context_zero_shot_ran101_nousedoc_f1.jsonl", "r")
    args = parser.parse_args()
    gold_data = load_dataset("wiki_qa", split="test")
    gold_answers_and_evidence = get_answers_and_evidence(gold_data)
    predicted_answers_and_evidence = {}
    for line in open(args.predictions):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            'question': prediction_data['question'],
            "answer": prediction_data["predicted_answer"],
            "evidence": ""
        }
    evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence,f1)
    print(json.dumps(evaluation_output, indent=2))


