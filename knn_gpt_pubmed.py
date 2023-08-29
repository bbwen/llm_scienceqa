import json
import time
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


from sklearn.model_selection import train_test_split


import os
import openai

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

openai.api_key = ('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')

def generate(prompt,max_tokens=1000, temperature=0.0):
    # tokens = tokenizer.tokenize(prompt)
    while True:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=temperature,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            time.sleep(1)
            break
        except Exception as e:
            time.sleep(5)
            continue
    return response["choices"][0]['text']

def chatgpt_generate(prompt,max_tokens=4096, temperature=0.0):
    while True:
        try:
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature = 0,
                    max_tokens = 256,
                    top_p=1,
                frequency_penalty = 0,
                    presence_penalty = 0
            )
        # print(response)
        # time.sleep(1)
            break
        except Exception as e:
            print(Exception)
            time.sleep(5)
            continue

    return response['choices'][0]['message']['content']


import random

random.seed(101)
kshot = 3
RANDOM = True
STATIC = True
#
# if STATIC:
#     static_sample = random.sample(train, k=kshot)
#     # print(static_sample)


def generate_explanation_answer(question, context, explained_knn_questions):

    # no context zero shot
    prompt = "For each example, create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\"，\"no\" and \"maybe\". Answer \"maybe\" when you are not sure about the answer\n\n"

    # zero-shot context qa
    # prompt = "For each example, use the documents to create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\"，\"no\" and \"maybe\". Answer \"maybe\"  when you are not sure about the answer\n\n"

    # if RANDOM:
    #     hits = random.sample(train, k=kshot)
    # elif STATIC:
    #     hits = static_sample
    # else:
    #     item_embeddings = model.encode(question)
    #
    #     all_top = util.dot_score(item_embeddings, train_embeddings)[0].topk(kshot)
    #
    #     hits = np.array(train)[all_top.indices].tolist()
    #
    #     hits.reverse()
    #
    for i, hit in enumerate(explained_knn_questions):
        prompt += "Example {0}:\n\n".format(i + 1)

        for j, d in enumerate(hit['evidence']):
            prompt += "[Document {0}]: {1}\n\n".format(j + 1, d)

        prompt += "Question: {0}\n\nExplanation: {1}\n\nAnswer: {2}\n\n".format(hit['question'], hit['explanation'],
                                                                                hit['answer'])


    # This prompt has 3 examples, each with a question and answer.
    # prompt = "For each example, use the documents to create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\", \"no\". Answer \"Unanswerable\" when not enough information is provided in the documents.\n\nExample 1:\n\nQuestion: Transgastric endoscopic splenectomy: is it possible?\n\nAnswer: yes.\n\nExample 2:\n\nQuestion: Does fluoridation reduce the use of dental services among adults?\n\nAnswer: maybe.\n\nExample 3:\n\nQuestion: Does strategy training reduce age-related deficits in working memory?\n\nAnswer: no.\n\n"


    # This prompt has 3 examples, each with a question, context, and answer.
    # prompt = "For each example, use the documents to create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\", \"no\" and \"maybe\".\n\nExample 1:\n\n[Document 1]: We have previously reported the feasibility of diagnostic and therapeutic peritoneoscopy including liver biopsy, gastrojejunostomy, and tubal ligation by an oral transgastric approach. We present results of per-oral transgastric splenectomy in a porcine model. The goal of this study was to determine the technical feasibility of per-oral transgastric splenectomy using a flexible endoscope. We performed acute experiments on 50-kg pigs. All animals were fed liquids for 3 days prior to procedure. The procedures were performed under general anesthesia with endotracheal intubation. The flexible endoscope was passed per orally into the stomach and puncture of the gastric wall was performed with a needle knife. The puncture was extended to create a 1.5-cm incision using a pull-type sphincterotome, and a double-channel endoscope was advanced into the peritoneal cavity. The peritoneal cavity was insufflated with air through the endoscope. The spleen was visualized. The splenic vessels were ligated with endoscopic loops and clips, and then mesentery was dissected using electrocautery. Endoscopic splenectomy was performed on six pigs. There were no complications during gastric incision and entrance into the peritoneal cavity. Visualization of the spleen and other intraperitoneal organs was very good. Ligation of the splenic vessels and mobilization of the spleen were achieved using commercially available devices and endoscopic accessories.\n\nQuestion: Transgastric endoscopic splenectomy: is it possible?\n\nAnswer: yes.\n\nExample 2:\n\n[Document 1]: The authors determine whether prevention influences the use of health services. Fluoridation's effect on restorative dental demand among 972 Washington state employees and spouses, aged 20 to 34 years, in two fluoridated communities and a nonfluoridated community was examined. At baseline, adults were interviewed by telephone, and oral assessments were conducted to measure personal characteristics, lifetime exposure to fluoridated water, oral disease, and the quality of restorations. Adults were followed for 2 years to measure dental demand from dental claims. Each adult's baseline and claims data were linked with provider and practice variables collected from the dentist who provided treatment. Relative to adults with no lifetime exposure to fluoridated water, adults drinking fluoridated water for half or more of their lives had less disease at baseline and a lower but nonsignificant probability of receiving a restoration in the follow-up period. In the 2-year follow-up period, however, more than half of the restorations were performed to replace fillings of satisfactory or ideal quality at baseline. When only teeth with decay and unsatisfactory fillings at baseline were considered, adults with high fluoridation exposure had a lower probability of receiving a restoration than adults with no exposure. Market effects also were detected in demand equations; relative to adults in the nonfluoridated community, adults residing in the fluoridated community with a large dentist supply received a greater number of restorations, suggesting potential supplier-induced demand from less disease and fewer patients.\n\nQuestion: Does fluoridation reduce the use of dental services among adults?\n\nAnswer: maybe.\n\n\n\nExample 3:\n\n[Document 1]: Older adults typically perform worse on measures of working memory (WM) than do young adults; however, age-related differences in WM performance might be reduced if older adults use effective encoding strategies. The purpose of the current experiment was to evaluate WM performance after training individuals to use effective encoding strategies. Participants in the training group (older adults: n = 39; young adults: n = 41) were taught about various verbal encoding strategies and their differential effectiveness and were trained to use interactive imagery and sentence generation on a list-learning task. Participants in the control group (older: n = 37; young: n = 38) completed an equally engaging filler task. All participants completed a pre- and post-training reading span task, which included self-reported strategy use, as well as two transfer tasks that differed in the affordance to use the trained strategies - a paired-associate recall task and the self-ordered pointing task. Both young and older adults were able to use the target strategies on the WM task and showed gains in WM performance after training. The age-related WM deficit was not greatly affected, however, and the training gains did not transfer to the other cognitive tasks. In fact, participants attempted to adapt the trained strategies for a paired-associate recall task, but the increased strategy use did not benefit their performance.\n\nQuestion: Does strategy training reduce age-related deficits in working memory?\n\nAnswer: no.\n\n"
    prompt += "Example {0}:\n\n".format(4)
    #
    # for k, document in enumerate(documents):
    #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document['title'])

    for k, document in enumerate(context):
        prompt += "[Document {0}]: {1}\n\n".format(k + 1, document)

    # for k, document in enumerate(entity_desc):
    #     prompt += "[Entity {0}]: {1}\n\n".format(k + 1, document)

    prompt += "Question: {0}\n\nExplanation:".format(question)

    tokens = tokenizer.tokenize(prompt)
    print(prompt)
    res = generate(prompt)

    if "answer:" not in res.lower():
        prompt = prompt + res + "\n\nAnswer:"
        res = res + "\n\nAnswer:" + generate(prompt)

    explanation = res.lower().split("answer:")[0]

    answer = res.lower().split("answer:")[1]

    return explanation, answer, prompt, res


import re
from tqdm import tqdm
import numpy as np
from datetime import date
import random

RANDOM = True

kshot = 3
docs = 3

regex = r"\[document \d+\]"

f = open("pubmed_gpt_knn_randomcontext_new.jsonl", "w")
data = open("../pubmedqa_hf/test.json", "r")
background = {}
pubmed_knn_test_dict = json.load(open("pubmed_gpt_knn_test.json"))

for line in open("../chatgpt_background_pubmed_100.jsonl", "r"):
    each = json.loads(line)
    background[each["question_id"]] = each["background"]

contriever_question =open("/Users/wenbingbing/PycharmProjects/qasper-led-baseline/pubmed/contriever/pubmed_contriever.jsonl")
contriever_question_dict = {}
for each in contriever_question:
    each = json.loads(each)
    contriever_question_dict[each['question_id']] = each['ctxs'][0]['text']
stratfy_by = []
all = []
context_all = []
for line in data:
    each = json.loads(line)
    all.append(each)
    question_id = each['id']
    question = each["sentence1"]
    context = each["sentence2"]
    context_all.append([context])
    answer = each["label"]
    if answer == "yes":
        stratfy_by.append(0)
    elif answer == "no":
        stratfy_by.append(1)
    elif answer == "maybe":
        stratfy_by.append(2)

X_train, test = train_test_split(all, test_size=0.5, random_state=42, stratify=stratfy_by)
print(len(test))
context_length = []
for each in tqdm(test):
    question_id = each['id']
    question = each["sentence1"]
    context = [each["sentence2"]]
    while True:
        random_sample = random.sample(context_all, k=1)
        if random_sample != context:
            break
    gt = each["label"]
    context_background = context + [background[question_id]]
    context_length.append(len(" ".join(context).split()))
    retrieval = contriever_question_dict[question_id]
    all_evidence_retrieval = context + [retrieval]
    context_random_append = context + random_sample
    explained_knn_questions = pubmed_knn_test_dict[question_id]

    explanation, answer, prompt, res = generate_explanation_answer(question, random_sample,explained_knn_questions)
    f.write(json.dumps({
        'question_id': question_id,
        'question': question,
        'predicted_answer': answer.rstrip(),
        'gt': gt,
        'explanation': explanation
    })+"\n")

# print("context_length {}".format(np.mean(context_length)))
#
#     f_log.write(json.dumps({
#         'question_id': item['question_id'],
#         "prompt": prompt
#     })+"\n")
