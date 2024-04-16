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


def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    if text_evidence_only:
                        evidence = [text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text]
                    else:
                        evidence = answer_info["evidence"]
                    references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def extract_answer(predict_answer):
    predict_answer = predict_answer.replace("</s>","")
    predict_answer = predict_answer.replace("<pad>","")
    predict_answer = predict_answer.replace("/n","")
    predict_answer = predict_answer.replace("\\","")
    if "unanswerable" in predict_answer.lower():
        predict_answer = "unanswerable"
    elif predict_answer.lower().startswith("yes") or predict_answer.lower().startswith(" yes"):
        predict_answer = "yes"
    elif predict_answer.lower().startswith("no") or predict_answer.lower().startswith(" no"):
        predict_answer = "no"
    # print(predict_answer)
    return predict_answer


def evaluate(gold, predicted,predicted_after,comp_file):
    hashas = 0.0
    nono = 0.0
    hasno = 0.0
    nohas = 0.0
    same = 0.0
    has_answer = 0.0
    max_answer_f1s = []
    max_evidence_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    unanswerable_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    max_answer_f1s_after = []
    max_answer_f1s_by_type_after = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    unanswerable_by_type_after = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            # max_answer_f1s.append(0.0)
            # max_evidence_f1s.append(0.0)
            continue
        answer = extract_answer(predicted[question_id]["answer"])
        answer_after = extract_answer(predicted_after[question_id]["answer"])

        answer_f1s_and_types = [
            (token_f1_score(answer, reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]
        answer_f1s_and_types_after = [
            (token_f1_score(answer_after, reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]


        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        max_answer_f1_after, answer_type_after = sorted(answer_f1s_and_types_after, key=lambda x: x[0], reverse=True)[0]
        if answer_type != "none":
            has_answer +=1


        if "unanswerable" in answer.lower():
            unanswerable_by_type[answer_type].append(1.0)
        else:
            unanswerable_by_type[answer_type].append(0.0)


        if "unanswerable" in answer_after.lower():
            unanswerable_by_type_after[answer_type_after].append(1.0)
        else:
            unanswerable_by_type_after[answer_type_after].append(0.0)


        # f1.write(json.dumps({
        #     'question_id': question_id,
        #     'predicted_answer': answer,
        #     'gt': [each['answer'] for each in gold[question_id]],
        #     'f1':answer_f1s_and_types
        # }) + "\n")
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        # print(type(gold[question_id][0]["evidence"]))
        # print(type(predicted[question_id]["evidence"]))
        max_answer_f1s_after.append(max_answer_f1_after)
        max_answer_f1s_by_type_after[answer_type_after].append(max_answer_f1_after)

        if answer != "unanswerable" and  answer_after == answer:
            same +=1
        if  answer != "unanswerable" and  answer_after != "unanswerable":
            hashas +=1
        elif  answer != "unanswerable" and answer_after == "unanswerable" :
            hasno +=1
        elif  answer == "unanswerable" and answer_after != "unanswerable" :
            nohas +=1
        elif  answer == "unanswerable" and answer_after == "unanswerable" :
            nono +=1

        comp_file.write(json.dumps({
        'question_id': question_id,
        'question': predicted[question_id]["question"],
        'after': answer_after,
        'before':answer
    })+"\n")

        evidence_f1s = [
            paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
            for reference in gold[question_id]
        ]
        max_evidence_f1s.append(max(evidence_f1s))

    mean = lambda x: sum(x) / len(x) if x else 0.0
    all = []
    for i in list(unanswerable_by_type.values()):
        all += i

    all_after = []
    for i in list(unanswerable_by_type_after.values()):
        all_after += i

    print("same",same/has_answer)
    print("hashas",hashas/len(all))
    print("hasno", hasno/len(all))
    print("nohas", nohas/len(all))
    print("nono", nono/len(all))

    has_all = []
    has_all_after = []
    for key, value in unanswerable_by_type.items():
        if key != "none":
            has_all += value

    for key, value in unanswerable_by_type_after.items():
        if key != "none":
            has_all_after += value
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Answerability_all": mean(all),
        "Answerability_has": mean(has_all),
        "noans F1 by type": {key: mean(value) for key, value in unanswerable_by_type.items()},

        "Answer F1 after": mean(max_answer_f1s_after),
        "Answer F1 by type after": {key: mean(value) for key, value in max_answer_f1s_by_type_after.items()},
        "Answerability_all_after": mean(all_after),
        "Answerability_has_after": mean(has_all_after),
        "Noans after": {key: mean(value) for key, value in unanswerable_by_type_after.items()},
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
    parser.add_argument(
        "--predictions_after",
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
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset"
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1"
    )
    # f1 = open("qasper_gpt_double_top_p1_given_true_evidence0_f1.jsonl", "r")
    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, args.text_evidence_only)
    predicted_answers_and_evidence = {}
    comp_file = open(args.comp_file, "w")
    predicted_answers_and_evidence_after = {}
    for line in open(args.predictions):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            'question': prediction_data['question'],
            "answer": prediction_data["predicted_answer"],
            "evidence": ""
        }
    for line in open(args.predictions_after):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence_after[prediction_data["question_id"]] = {
            'question': prediction_data['question'],
            "answer": prediction_data["predicted_answer"],
            "evidence": ""
        }
    evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence,predicted_answers_and_evidence_after,comp_file)

    print(evaluation_output)


