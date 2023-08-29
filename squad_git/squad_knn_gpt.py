import json
import time
from transformers import GPT2Tokenizer
import random
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import load_dataset

random.seed(101)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

questions = {}
abstract_dict = {}
title_dict = {}
intro_dict = {}



def random_pertube(full_text, evidence, n=1, attempt=1):
    pertubations = []
    for i in range(n):
        section = random.sample(full_text, k=1)
        # print(section)
        paragraph = random.sample(section[0]['paragraphs'], k=1)
        if paragraph[0] not in evidence and paragraph[0] != "":
            pertubations.append(paragraph[0])
        else:
            pertubations = pertubations + random_pertube(full_text, evidence, n=1)

    if len(pertubations) == 0 and attempt < 5:
        return random_pertube(full_text, evidence, n, attempt + 1)

    return pertubations


def generate_knn_train(question_id):
    explained_questions = []
    knn_dict = {}
    qasper_train_dict = {}
    qasper_train = json.load(open('/Users/wenbingbing/Downloads/qasper-train-dev-v0.3/qasper-train-v0.3.json', 'r'))
    knn_result = "/Users/wenbingbing/PycharmProjects/qasper-led-baseline/ColBERT/qasper_knn.jsonl"
    for k in list(qasper_train.keys()):
        for qa in qasper_train[k]['qas']:
            qasper_train_dict[qa['question_id']] =[qa['question'], qa['answers'][0]['answer'], qa['answers'][0]['answer']['evidence'], qasper_train[k]['title'], qasper_train[k]['abstract'], qasper_train[k]['full_text']]

    for line in open(knn_result):
        each = json.loads(line)
        knn_dict[each['question_id']] = each['retrieved id']

    for each in knn_dict[question_id]:
        prompt = "For each example, explain how each document is used to answer the question:\n\nExample 1:\n\n[Document 1]: In this section we describe a number of experiments targeted to compare the performance of popular named entity recognition algorithms on our data. We trained and evaluated Stanford NER, spaCy 2.0, and a recurrent model similar to BIBREF13 , BIBREF14 that uses bidirectional LSTM cells for character-based feature extraction and CRF, described in Guillaume Genthial's Sequence Tagging with Tensorflow blog post BIBREF15 .\n\n[Document 2]: Stanford NER is conditional random fields (CRF) classifier based on lexical and contextual features such as the current word, character-level n-grams of up to length 6 at its beginning and the end, previous and next words, word shape and sequence features BIBREF16 .\n\n[Document 3]: spaCy 2.0 uses a CNN-based transition system for named entity recognition. For each token, a Bloom embedding is calculated based on its lowercase form, prefix, suffix and shape, then using residual CNNs, a contextual representation of that token is extracted that potentially draws information from up to 4 tokens from each side BIBREF17 . Each update of the transition system's configuration is a classification task that uses the contextual representation of the top token on the stack, preceding and succeeding tokens, first two tokens of the buffer, and their leftmost, second leftmost, rightmost, second rightmost children. The valid transition with the highest score is applied to the system. This approach reportedly performs within 1% of the current state-of-the-art for English . In our experiments, we tried out 50-, 100-, 200- and 300-dimensional pre-trained GloVe embeddings. Due to time constraints, we did not tune the rest of hyperparameters and used their default values.\n\n[Document 4]: In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are from political, sports, local and world news (Figures FIGREF8 , FIGREF9 ), covering the period between August 2012 and July 2018. The dataset provides annotations for 3 popular named entity classes: people (PER), organizations (ORG), and locations (LOC), and is released in CoNLL03 format with IOB tagging scheme. Tokens and sentences were segmented according to the UD standards for the Armenian language BIBREF11 .\n\n[Document 5]: The main model that we focused on was the recurrent model with a CRF top layer, and the above-mentioned methods served mostly as baselines. The distinctive feature of this approach is the way contextual word embeddings are formed. For each token separately, to capture its word shape features, character-based representation is extracted using a bidirectional LSTM BIBREF18 . This representation gets concatenated with a distributional word vector such as GloVe, forming an intermediate word embedding. Using another bidirectional LSTM cell on these intermediate word embeddings, the contextual representation of tokens is obtained (Figure FIGREF17 ). Finally, a CRF layer labels the sequence of these contextual representations. In our experiments, we used Guillaume Genthial's implementation of the algorithm. We set the size of character-based biLSTM to 100 and the size of second biLSTM network to 300\n\nQuestion: what ner models were evaluated?\n\nAnswer: Stanford NER algorithm, the spaCy 2.0 algorithm, recurrent model with a CRF top layer.\n\nExplanation: According to [Document 1], the Stanford NER algorithm, the spaCy 2.0 algorithm, and a recurrent model with a CRF top layer were evaluated. This information is further supported by [Document 2], [Document 3], and [Document 5].\n\nExample 2:\n\n"
        question = qasper_train_dict[each][0]
        answer_info = qasper_train_dict[each][1]
        evidence = qasper_train_dict[each][2]
        full_text = qasper_train_dict[each][5]
        if answer_info["unanswerable"]:
            answer = "Unanswerable"
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

        evidence = [text for text in evidence if "FLOAT SELECTED" not in text]

        # pertubed_evidence = evidence + random_pertube(full_text, evidence)
        #
        # random.shuffle(pertubed_evidence)
        # use only evidence
        for i, d in enumerate(evidence):
            prompt += "[Document {0}]: {1}\n\n".format(i + 1, d)
        prompt += "Question: {0}\n\nAnswer: {1}\n\nExplanation:".format(question, answer)
        res = generate(prompt)
        explained_questions.append({
            "question": question,
            "evidence": evidence,
            "answer": answer,
            "explanation": res
        })
    return explained_questions


def sample4test(data_dict):
    stratfy_by = []
    data =[]
    for key in data_dict.keys():
        if data_dict[key][0]['answer'] == []:
            stratfy_by.append(0)
            data.append(key)
        else:
            stratfy_by.append(1)
            data.append(key)
    X_train, test, y_train, y_test = train_test_split(data, stratfy_by, test_size=0.02, random_state=42, stratify=stratfy_by)
    print(len(test))
    return test



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
            break
        # print(response)
        except Exception as e:
            print(Exception)
            time.sleep(5)
            continue
    return response['choices'][0]['message']['content']

# kshot = 3
# RANDOM = True
# STATIC = True
# if STATIC:
#     static_sample = random.sample(train, k=kshot)
#     # print(static_sample)

def generate_explanation_answer(question, conetxt, explained_knn_questions):
    # zero-shot no context qa
    prompt = "For each example, create an shortest \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer.\n\n"

    #zero-shot context qa
    # prompt = "For each example, use the documents to create a shortest \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer.\n\n"

    # prompt = "For each example, use the documents to create an shortest \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"\" when you are not sure about the answer.\n\n"

    #zero-shot context qa unanswerable changed to ''
    # prompt = "For each example, use the documents to create an shortest \"Answer\" to the \"Question\". Answer \"Unanswrable\" when not enough information is provided in the documents.\n\n"


    # intructions of using the context,
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"

    for i, hit in enumerate(explained_knn_questions):
        prompt += "Example {0}:\n\n".format(i + 1)

        for j, d in enumerate(hit['evidence']):
            prompt += "[Document {0}]: {1}\n\n".format(j + 1, d)
        if hit['answer'] == "":
            gt_answer = "Unanswerable"
        else :
            gt_answer = hit['answer']

        prompt += "Question: {0}\n\nExplanation: {1}\n\nAnswer: {2}\n\n".format(hit['question'], hit['explanation'],gt_answer)



    # This prompt has 4 examples, each with a question, explanation, and answer including unanswerable .
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\nExample 1:\n\n[Document 1]: In Figure FIGREF8 , we see that overall sentiment averages rarely show movement post-event: that is, only Hurricane Florence shows a significant difference in average tweet sentiment pre- and post-event at the 1% level, corresponding to a 0.12 point decrease in positive climate change sentiment. However, controlling for the same group of users tells a different story: both Hurricane Florence and Hurricane Michael have significant tweet sentiment average differences pre- and post-event at the 1% level. Within-cohort, Hurricane Florence sees an increase in positive climate change sentiment by 0.21 points, which is contrary to the overall average change (the latter being likely biased since an influx of climate change deniers are likely to tweet about hurricanes only after the event). Hurricane Michael sees an increase in average tweet sentiment of 0.11 points, which reverses the direction of tweets from mostly negative pre-event to mostly positive post-event. Likely due to similar bias reasons, the Mendocino wildfires in California see a 0.06 point decrease in overall sentiment post-event, but a 0.09 point increase in within-cohort sentiment. Methodologically, we assert that overall averages are not robust results to use in sentiment analyses.\n\n[Document 2]: The second data batch consists of event-related tweets for five natural disasters occurring in the U.S. in 2018. These are: the East Coast Bomb Cyclone (Jan. 2 - 6); the Mendocino, California wildfires (Jul. 27 - Sept. 18); Hurricane Florence (Aug. 31 - Sept. 19); Hurricane Michael (Oct. 7 - 16); and the California Camp Fires (Nov. 8 - 25). For each disaster, we scraped tweets starting from two weeks prior to the beginning of the event, and continuing through two weeks after the end of the event. Summary statistics on the downloaded event-specific tweets are provided in Table TABREF1 . Note that the number of tweets occurring prior to the two 2018 sets of California fires are relatively small. This is because the magnitudes of these wildfires were relatively unpredictable, whereas blizzards and hurricanes are often forecast weeks in advance alongside public warnings. The first (influential tweet data) and second (event-related tweet data) batches are de-duplicated to be mutually exclusive. In Section SECREF2 , we perform geographic analysis on the event-related tweets from which we can scrape self-reported user city from Twitter user profile header cards; overall this includes 840 pre-event and 5,984 post-event tweets.\n\nQuestion: Which five natural disasters were examined?\n\nExplanation:  The five natural disasters examined in [Document 2] are the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires. This information is further supported by [Document 1].\n\nAnswer: the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires\n\nExample 2:\n\n[Document 1]: N-GrAM ranked first in all cases except for the language variety task. In this case, the baseline was the top-ranked system, and ours was second by a small margin. Our system significantly out-performed the baseline on the joint task, as the baseline scored significantly lower for the gender task than for the variety task.\n\n[Document 2]: ()\n\nQuestion: On which task does do model do worst?\n\nExplanation: \n\nAccording to [Document 1], the model does worst on the gender prediction task.\n\nAnswer: Gender prediction task\n\nExample 3:\n\n[Document 1]: As seen in Table TABREF38, both SMERTI variations achieve higher STES and outperform the other models overall, with the WordNet models performing the worst. SMERTI excels especially on fluency and content similarity. The transformer variation achieves slightly higher SLOR, while the RNN variation achieves slightly higher CSS. The WordNet models perform strongest in sentiment preservation (SPA), likely because they modify little of the text and only verbs and nouns. They achieve by far the lowest CSS, likely in part due to this limited text replacement. They also do not account for context, and many words (e.g. proper nouns) do not exist in WordNet. Overall, the WordNet models are not very effective at STE.\n\n[Document 2]: We evaluate on three datasets: Yelp and Amazon reviews BIBREF1, and Kaggle news headlines BIBREF2. We implement three baseline models for comparison: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), and Word2Vec Semantic Text Exchange Model (W2V-STEM).\n\nQuestion: What are the baseline models mentioned in the paper?\n\nExplanation:  The baseline models are NWN-STEM, GWN-STEM, and W2V-STEM. This is mentioned in [Document 2].\n\nAnswer: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), Word2Vec Semantic Text Exchange Model (W2V-STEM)\n\nExample 4:\n\n[Document 1]: ()\n\nQuestion: Are there privacy concerns with clinical data?\n\nExplanation: \n\nThe documents don't give enough information to answer the question.\n\nAnswer: unanswerable\n\nExample 5:\n\n"

    # This prompt has 4 examples, each with a kind of question type.
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\nExample 1:\n\n[Document 1]: N-GrAM ranked first in all cases except for the language variety task. In this case, the baseline was the top-ranked system, and ours was second by a small margin. Our system significantly out-performed the baseline on the joint task, as the baseline scored significantly lower for the gender task than for the variety task.\n\n[Document 2]: ()\n\nQuestion: On which task does do model do worst?\n\nExplanation: \n\nAccording to [Document 1], the model does worst on the gender prediction task.\n\nAnswer: Gender prediction task\n\nExample 2:\n\n[Document 1]: As seen in Table TABREF38, both SMERTI variations achieve higher STES and outperform the other models overall, with the WordNet models performing the worst. SMERTI excels especially on fluency and content similarity. The transformer variation achieves slightly higher SLOR, while the RNN variation achieves slightly higher CSS. The WordNet models perform strongest in sentiment preservation (SPA), likely because they modify little of the text and only verbs and nouns. They achieve by far the lowest CSS, likely in part due to this limited text replacement. They also do not account for context, and many words (e.g. proper nouns) do not exist in WordNet. Overall, the WordNet models are not very effective at STE.\n\n[Document 2]: We evaluate on three datasets: Yelp and Amazon reviews BIBREF1, and Kaggle news headlines BIBREF2. We implement three baseline models for comparison: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), and Word2Vec Semantic Text Exchange Model (W2V-STEM).\n\nQuestion: What are the baseline models mentioned in the paper?\n\nExplanation:  The baseline models are NWN-STEM, GWN-STEM, and W2V-STEM. This is mentioned in [Document 2].\n\nAnswer: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), Word2Vec Semantic Text Exchange Model (W2V-STEM)\n\nExample 3:\n\n[Document 1]: ()\n\nQuestion: How better is performance of natural language based agents in experiments?\n\nExplanation: \n\nThe documents don't give enough information to answer the question.\n\nAnswer: unanswerable\n\nExample 4:\n\n[Document 1]:To evaluate this application, we inject random triples into the graph, and measure the ability of to detect the errors using our optimization. We consider two types of incorrect triples: 1) incorrect triples in the form of $\\langle s^{\\prime }, r, o\\rangle $ where $s^{\\prime }$ is chosen randomly from all of the entities, and 2) incorrect triples in the form of $\\langle s^{\\prime }, r^{\\prime }, o\\rangle $ where $s^{\\prime }$ and $r^{\\prime }$ are chosen randomly. We choose 100 random triples from the observed graph, and for each of them, add an incorrect triple (in each of the two scenarios) to its neighborhood. Then, after retraining DistMult on this noisy training data, we identify error triples through a search over the neighbors of the 100 facts. The result of choosing the neighbor with the least influence on the target is provided in the Table 7 . When compared with baselines that randomly choose one of the neighbors, or assume that the fact with the lowest score is incorrect, we see that outperforms both of these with a considerable gap, obtaining an accuracy of $42\\%$ and $55\\%$ in detecting errors.\n\nQuestion: Can this adversarial approach be used to directly improve model accuracy?n\nExplanation: \n\n [Document 1] states that the adversarial approach can be used to directly improve model accuracy. This is supported by the fact that the approach outperforms both of the baselines with a considerable gap.\n\nAnswer: Yes\n\n"


    # This prompt has 3 examples, each with a question, explanation, and answer.
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Pay attention to answer only \"yes\" or \"no\" in boolean questions. Answer \"Unanswerable\" when not enough information is provided in the documents. \n\nExample 1:\n\n[Document 1]: In Figure FIGREF8 , we see that overall sentiment averages rarely show movement post-event: that is, only Hurricane Florence shows a significant difference in average tweet sentiment pre- and post-event at the 1% level, corresponding to a 0.12 point decrease in positive climate change sentiment. However, controlling for the same group of users tells a different story: both Hurricane Florence and Hurricane Michael have significant tweet sentiment average differences pre- and post-event at the 1% level. Within-cohort, Hurricane Florence sees an increase in positive climate change sentiment by 0.21 points, which is contrary to the overall average change (the latter being likely biased since an influx of climate change deniers are likely to tweet about hurricanes only after the event). Hurricane Michael sees an increase in average tweet sentiment of 0.11 points, which reverses the direction of tweets from mostly negative pre-event to mostly positive post-event. Likely due to similar bias reasons, the Mendocino wildfires in California see a 0.06 point decrease in overall sentiment post-event, but a 0.09 point increase in within-cohort sentiment. Methodologically, we assert that overall averages are not robust results to use in sentiment analyses.\n\n[Document 2]: The second data batch consists of event-related tweets for five natural disasters occurring in the U.S. in 2018. These are: the East Coast Bomb Cyclone (Jan. 2 - 6); the Mendocino, California wildfires (Jul. 27 - Sept. 18); Hurricane Florence (Aug. 31 - Sept. 19); Hurricane Michael (Oct. 7 - 16); and the California Camp Fires (Nov. 8 - 25). For each disaster, we scraped tweets starting from two weeks prior to the beginning of the event, and continuing through two weeks after the end of the event. Summary statistics on the downloaded event-specific tweets are provided in Table TABREF1 . Note that the number of tweets occurring prior to the two 2018 sets of California fires are relatively small. This is because the magnitudes of these wildfires were relatively unpredictable, whereas blizzards and hurricanes are often forecast weeks in advance alongside public warnings. The first (influential tweet data) and second (event-related tweet data) batches are de-duplicated to be mutually exclusive. In Section SECREF2 , we perform geographic analysis on the event-related tweets from which we can scrape self-reported user city from Twitter user profile header cards; overall this includes 840 pre-event and 5,984 post-event tweets. \n\nQuestion: Which five natural disasters were examined?\n\nExplanation:  The five natural disasters examined in [Document 2] are the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires. This information is further supported by [Document 1].\n\nAnswer: the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires\n\nExample 2:\n\n[Document 1]: N-GrAM ranked first in all cases except for the language variety task. In this case, the baseline was the top-ranked system, and ours was second by a small margin. Our system significantly out-performed the baseline on the joint task, as the baseline scored significantly lower for the gender task than for the variety task.\n\n[Document 2]: ()\n\nQuestion: On which task does do model do worst?\n\nExplanation: \n\nAccording to [Document 1], the model does worst on the gender prediction task.\n\nAnswer: Gender prediction task\n\nExample 3:\n\n[Document 1]: As seen in Table TABREF38, both SMERTI variations achieve higher STES and outperform the other models overall, with the WordNet models performing the worst. SMERTI excels especially on fluency and content similarity. The transformer variation achieves slightly higher SLOR, while the RNN variation achieves slightly higher CSS. The WordNet models perform strongest in sentiment preservation (SPA), likely because they modify little of the text and only verbs and nouns. They achieve by far the lowest CSS, likely in part due to this limited text replacement. They also do not account for context, and many words (e.g. proper nouns) do not exist in WordNet. Overall, the WordNet models are not very effective at STE.\n\n[Document 2]: We evaluate on three datasets: Yelp and Amazon reviews BIBREF1, and Kaggle news headlines BIBREF2. We implement three baseline models for comparison: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), and Word2Vec Semantic Text Exchange Model (W2V-STEM).\n\nQuestion: What are the baseline models mentioned in the paper?\n\nExplanation:  The baseline models are NWN-STEM, GWN-STEM, and W2V-STEM. This is mentioned in [Document 2].\n\nAnswer: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), Word2Vec Semantic Text Exchange Model (W2V-STEM)\n\nExample 4:\n\n"

    prompt += "Example {0}:\n\n".format(4)
    #
    # for k, document in enumerate(documents):
    #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document['title'])

    for k, document in enumerate(conetxt):
        prompt += "[Document {0}]: {1}\n\n".format(k + 1, document)

    # for k, document in enumerate(entity_desc):
    #     prompt += "[Entity {0}]: {1}\n\n".format(k + 1, document)

    prompt += "Question: {0}\n\nExplanation:".format(question)
    # prompt += "Question: {0}\n\n:".format(question)

    tokens = tokenizer.tokenize(prompt)
    # print(prompt)
    res = generate(prompt)
    print(res)

    if "answer:" not in res.lower():
        prompt = prompt + res + "\n\nAnswer:"
        res = res + "\n\nAnswer:" + generate(prompt)

    explanation = res.lower().split("answer:")[0]

    answer = res.lower().split("answer:")[1]

    return explanation, answer, prompt, res



import numpy as np
from datetime import date
import random

RANDOM = True

kshot = 3
docs = 3

if __name__ == '__main__':

    valid_dataset = load_dataset("squad_v2", split="validation")
    print(len(valid_dataset))
    valid_dict = {}
    context_all = []
    for each in valid_dataset:
        # print(each)
        context_all.append(each['context'])
        valid_dict[each['id']] = [{'context': each['context'], 'question': each['question'],'answer': each['answers']['text']}]
    print(len(valid_dict.keys()))
    sample_valid= sample4test(valid_dict)
    # print(sample_valid)

    f1 = open("1_squad2_gpt_knn_nocontext_ran101_new.jsonl", 'w')
    # f2 = open("2_squad2_gpt_nocontext_zero_shot_ran101.jsonl",'w')=

    f_log = open("1_squad2_gpt_knn_randomcontext_ran101_log.jsonl", 'w')
    logs = []
    tokens= []
    num = 0
    squad2_knn_test_dict = json.load(open("../squad_gpt_knn_test.json",'r'))

    # q_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_q_entity_desc_test.json"))
    # evi_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_evi_entity_desc_test.json"))
    for item in tqdm(sample_valid):
        question = valid_dict[item][0]['question']
        context = valid_dict[item][0]['context']
        gold_answer = valid_dict[item][0]['answer']
        explained_knn_questions = squad2_knn_test_dict[item]

        while True:
            random_sample = random.sample(context_all, k=1)
            if random_sample != context:
                break

        explanation, answer, prompt, res = generate_explanation_answer(question, [""], explained_knn_questions)

        f1.write(json.dumps({
            'question_id': item,
            'question': question,
            'predicted_answer': answer.rstrip(),
            'gold_answer': gold_answer,
            'predicted_evidence': context,
            'explanation': explanation
        })+"\n")

        # f2.write(json.dumps({
        #     'question_id': item,
        #     'question': question,
        #     'predicted_answer': answer1.rstrip(),
        #     'predicted_evidence': context,
        #     'explanation': explanation1
        # })+"\n")

        f_log.write(json.dumps({
            'question_id': item,
            "prompt": prompt
        })+"\n")
