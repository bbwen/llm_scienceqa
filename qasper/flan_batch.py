import json
import time
from transformers import GPT2Tokenizer
api_key="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" #Hugging face API
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to("cuda")
qas = json.load(open("qasper_reranked_section_title.json",'r'))
data = json.load(open("qasper-test-v0.3.json", "r"))
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


# def generate_knn_train(question_id):
#     explained_questions = []
#     knn_dict = {}
#     qasper_train_dict = {}
#     qasper_train = json.load(open('qasper-train-v0.3.json', 'r'))
#     knn_result = "/home/ec2-user/qasper_knn.jsonl"
#     for k in list(qasper_train.keys()):
#         for qa in qasper_train[k]['qas']:
#             qasper_train_dict[qa['question_id']] =[qa['question'], qa['answers'][0]['answer'], qa['answers'][0]['answer']['evidence'], qasper_train[k]['title'], qasper_train[k]['abstract'], qasper_train[k]['full_text']]
# 
#     for line in open(knn_result):
#         each = json.loads(line)
#         knn_dict[each['question_id']] = each['retrieved id']
# 
#     for each in knn_dict[question_id]:
#         prompt = "For each example, explain how each document is used to answer the question:\n\nExample 1:\n\n[Document 1]: In this section we describe a number of experiments targeted to compare the performance of popular named entity recognition algorithms on our data. We trained and evaluated Stanford NER, spaCy 2.0, and a recurrent model similar to BIBREF13 , BIBREF14 that uses bidirectional LSTM cells for character-based feature extraction and CRF, described in Guillaume Genthial's Sequence Tagging with Tensorflow blog post BIBREF15 .\n\n[Document 2]: Stanford NER is conditional random fields (CRF) classifier based on lexical and contextual features such as the current word, character-level n-grams of up to length 6 at its beginning and the end, previous and next words, word shape and sequence features BIBREF16 .\n\n[Document 3]: spaCy 2.0 uses a CNN-based transition system for named entity recognition. For each token, a Bloom embedding is calculated based on its lowercase form, prefix, suffix and shape, then using residual CNNs, a contextual representation of that token is extracted that potentially draws information from up to 4 tokens from each side BIBREF17 . Each update of the transition system's configuration is a classification task that uses the contextual representation of the top token on the stack, preceding and succeeding tokens, first two tokens of the buffer, and their leftmost, second leftmost, rightmost, second rightmost children. The valid transition with the highest score is applied to the system. This approach reportedly performs within 1% of the current state-of-the-art for English . In our experiments, we tried out 50-, 100-, 200- and 300-dimensional pre-trained GloVe embeddings. Due to time constraints, we did not tune the rest of hyperparameters and used their default values.\n\n[Document 4]: In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are from political, sports, local and world news (Figures FIGREF8 , FIGREF9 ), covering the period between August 2012 and July 2018. The dataset provides annotations for 3 popular named entity classes: people (PER), organizations (ORG), and locations (LOC), and is released in CoNLL03 format with IOB tagging scheme. Tokens and sentences were segmented according to the UD standards for the Armenian language BIBREF11 .\n\n[Document 5]: The main model that we focused on was the recurrent model with a CRF top layer, and the above-mentioned methods served mostly as baselines. The distinctive feature of this approach is the way contextual word embeddings are formed. For each token separately, to capture its word shape features, character-based representation is extracted using a bidirectional LSTM BIBREF18 . This representation gets concatenated with a distributional word vector such as GloVe, forming an intermediate word embedding. Using another bidirectional LSTM cell on these intermediate word embeddings, the contextual representation of tokens is obtained (Figure FIGREF17 ). Finally, a CRF layer labels the sequence of these contextual representations. In our experiments, we used Guillaume Genthial's implementation of the algorithm. We set the size of character-based biLSTM to 100 and the size of second biLSTM network to 300\n\nQuestion: what ner models were evaluated?\n\nAnswer: Stanford NER algorithm, the spaCy 2.0 algorithm, recurrent model with a CRF top layer.\n\nExplanation: According to [Document 1], the Stanford NER algorithm, the spaCy 2.0 algorithm, and a recurrent model with a CRF top layer were evaluated. This information is further supported by [Document 2], [Document 3], and [Document 5].\n\nExample 2:\n\n"
#         question = qasper_train_dict[each][0]
#         answer_info = qasper_train_dict[each][1]
#         evidence = qasper_train_dict[each][2]
#         full_text = qasper_train_dict[each][5]
#         if answer_info["unanswerable"]:
#             answer = "Unanswerable"
#         else:
#             if answer_info["extractive_spans"]:
#                 answer = ", ".join(answer_info["extractive_spans"])
#                 answer_type = "extractive"
#             elif answer_info["free_form_answer"]:
#                 answer = answer_info["free_form_answer"]
#                 answer_type = "abstractive"
#             elif answer_info["yes_no"]:
#                 answer = "Yes"
#                 answer_type = "boolean"
#             elif answer_info["yes_no"] is not None:
#                 answer = "No"
#                 answer_type = "boolean"
# 
#         evidence = [text for text in evidence if "FLOAT SELECTED" not in text]
# 
#         # pertubed_evidence = evidence + random_pertube(full_text, evidence)
#         #
#         # random.shuffle(pertubed_evidence)
#         # use only evidence
#         for i, d in enumerate(evidence):
#             prompt += "[Document {0}]: {1}\n\n".format(i + 1, d)
#         prompt += "Question: {0}\n\nAnswer: {1}\n\nExplanation:".format(question, answer)
#         res = generate(prompt)
#         explained_questions.append({
#             "question": question,
#             "evidence": evidence,
#             "answer": answer,
#             "explanation": res
#         })
#     return explained_questions






for k in data:
    for qa in data[k]['qas']:
        questions[qa['question_id']] = qa
        abstract_dict[qa['question_id']] = data[k]['abstract']
        title_dict[qa['question_id']] = data[k]['title']
        intro_dict[qa['question_id']] = data[k]['full_text'][0]['paragraphs']


def decompose(question):
    prompt="Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.\n\nExample 1:\n\nQuestion: Is Hamlet more common on IMDB than Comedy of Errors?\n\nDecompositions: \n1: How many listings of Hamlet are there on IMDB?\n2: How many listing of Comedy of Errors is there on IMDB?\n\nExample 2:\n\nQuestion: Are birds important to badminton?\n\nDecompositions:\nThe question needs no decomposition\n\nExample 3:\n\nQuestion: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?\n\nDecompositions:\n1: What is the minimum driving age in the US?\n2: What is the minimum age for someone to be employed in the US?\n\nExample 4:\n\nQuestion: Are all cucumbers the same texture?\n\nDecompositions:\nThe question needs no decomposition\n\nExample 5:\n\nQuestion: Hydrogen's atomic number squared exceeds number of Spice Girls?\n\nDecompositions:\n1: What is the atomic number of hydrogen?\n2: How many Spice Girls are there?\n\nExample 6:\n\nQuestion: {0}\n\nDecompositions:"

    res = generate(prompt.format(question), max_tokens=256)
    # print(res)
    if res.lower().strip() == "the question needs no decomposition.":
        return [question]
    try:
        questions = [l for l in res.splitlines() if l != ""]
        questions = [q.split(':')[1].strip() for q in questions]
        return questions
    except:
        return [question]

extractives = []
abstractives = []
none_ = []
yes_no = []
stratfy_by = []
for item in qas:
    annotation_info = questions[item['question_id']]["answers"][0]
    # annotation_info = item['answers'][0]
    answer_info = annotation_info["answer"]
    if answer_info["unanswerable"]:
        stratfy_by.append(0)
        none_.append(item)
    else:
        if answer_info["extractive_spans"]:
            extractives.append(item)
            stratfy_by.append(1)
        elif answer_info["free_form_answer"]:
            # answer = answer_info["free_form_answer"]
            abstractives.append(item)
            stratfy_by.append(2)
            # answer_type = "abstractive"
        elif answer_info["yes_no"]:
            yes_no.append(item)
            stratfy_by.append(3)
        elif answer_info["yes_no"] is not None:
            yes_no.append(item)
            stratfy_by.append(3)
print(len(extractives))
print(len(abstractives))
print(len(none_))
print(len(yes_no))

# 732
# 371
# 139
# 209
# 218


from sklearn.model_selection import train_test_split
X_train, test, y_train, y_test = train_test_split(qas, stratfy_by, test_size=0.15, random_state=42, stratify=stratfy_by)
print(len(test))

#from sentence_transformers import SentenceTransformer, util
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

# train = json.load(open("explained_train_1_5.json",'r'))
# train = json.load(open("../explained_train_ori_evidence_5_50.json",'r'))
# train_embeddings = model.encode([a['question'] for a in train])

import os
import openai

#from transformers import GPT2Tokenizer
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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
        break

    return response["choices"][0]['text']

def chatgpt_generate(prompt,max_tokens=4096, temperature=0.0):
    while True:
        # try:
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
        print(response)
        time.sleep(1)
        break
        # except Exception as e:
        #     print(Exception)
        #     time.sleep(5)
        #     continue
        # break

    return response['choices'][0]['message']['content']


import random

random.seed(101)
kshot = 3
RANDOM = True
STATIC = True

def generate_explanation_answer(question, documents,evidence, abstract, entity_desc, explained_knn_questions):
    # zero-shot no context qa
    # prompt = "For each example, create an \"Answer\" and to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"
    # intructions of using the context,
    prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"

    for i, hit in enumerate(explained_knn_questions):
        prompt += "Example {0}:\n\n".format(i + 1)
        for j, d in enumerate(hit['evidence']):
            prompt += "[Document {0}]: {1}\n\n".format(j + 1, d)
        prompt += "Question: {0}\n\nExplanation: {1}\n\nAnswer: {2}\n\n".format(hit['question'], hit['explanation'], hit['answer'])
    import json
    import time
    from transformers import GPT2Tokenizer
    api_key = "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"  # Hugging face API
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to("cuda")
    qas = json.load(open("qasper_reranked_section_title.json", 'r'))
    data = json.load(open("qasper-test-v0.3.json", "r"))
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

    # def generate_knn_train(question_id):
    #     explained_questions = []
    #     knn_dict = {}
    #     qasper_train_dict = {}
    #     qasper_train = json.load(open('qasper-train-v0.3.json', 'r'))
    #     knn_result = "/home/ec2-user/qasper_knn.jsonl"
    #     for k in list(qasper_train.keys()):
    #         for qa in qasper_train[k]['qas']:
    #             qasper_train_dict[qa['question_id']] =[qa['question'], qa['answers'][0]['answer'], qa['answers'][0]['answer']['evidence'], qasper_train[k]['title'], qasper_train[k]['abstract'], qasper_train[k]['full_text']]
    #
    #     for line in open(knn_result):
    #         each = json.loads(line)
    #         knn_dict[each['question_id']] = each['retrieved id']
    #
    #     for each in knn_dict[question_id]:
    #         prompt = "For each example, explain how each document is used to answer the question:\n\nExample 1:\n\n[Document 1]: In this section we describe a number of experiments targeted to compare the performance of popular named entity recognition algorithms on our data. We trained and evaluated Stanford NER, spaCy 2.0, and a recurrent model similar to BIBREF13 , BIBREF14 that uses bidirectional LSTM cells for character-based feature extraction and CRF, described in Guillaume Genthial's Sequence Tagging with Tensorflow blog post BIBREF15 .\n\n[Document 2]: Stanford NER is conditional random fields (CRF) classifier based on lexical and contextual features such as the current word, character-level n-grams of up to length 6 at its beginning and the end, previous and next words, word shape and sequence features BIBREF16 .\n\n[Document 3]: spaCy 2.0 uses a CNN-based transition system for named entity recognition. For each token, a Bloom embedding is calculated based on its lowercase form, prefix, suffix and shape, then using residual CNNs, a contextual representation of that token is extracted that potentially draws information from up to 4 tokens from each side BIBREF17 . Each update of the transition system's configuration is a classification task that uses the contextual representation of the top token on the stack, preceding and succeeding tokens, first two tokens of the buffer, and their leftmost, second leftmost, rightmost, second rightmost children. The valid transition with the highest score is applied to the system. This approach reportedly performs within 1% of the current state-of-the-art for English . In our experiments, we tried out 50-, 100-, 200- and 300-dimensional pre-trained GloVe embeddings. Due to time constraints, we did not tune the rest of hyperparameters and used their default values.\n\n[Document 4]: In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are from political, sports, local and world news (Figures FIGREF8 , FIGREF9 ), covering the period between August 2012 and July 2018. The dataset provides annotations for 3 popular named entity classes: people (PER), organizations (ORG), and locations (LOC), and is released in CoNLL03 format with IOB tagging scheme. Tokens and sentences were segmented according to the UD standards for the Armenian language BIBREF11 .\n\n[Document 5]: The main model that we focused on was the recurrent model with a CRF top layer, and the above-mentioned methods served mostly as baselines. The distinctive feature of this approach is the way contextual word embeddings are formed. For each token separately, to capture its word shape features, character-based representation is extracted using a bidirectional LSTM BIBREF18 . This representation gets concatenated with a distributional word vector such as GloVe, forming an intermediate word embedding. Using another bidirectional LSTM cell on these intermediate word embeddings, the contextual representation of tokens is obtained (Figure FIGREF17 ). Finally, a CRF layer labels the sequence of these contextual representations. In our experiments, we used Guillaume Genthial's implementation of the algorithm. We set the size of character-based biLSTM to 100 and the size of second biLSTM network to 300\n\nQuestion: what ner models were evaluated?\n\nAnswer: Stanford NER algorithm, the spaCy 2.0 algorithm, recurrent model with a CRF top layer.\n\nExplanation: According to [Document 1], the Stanford NER algorithm, the spaCy 2.0 algorithm, and a recurrent model with a CRF top layer were evaluated. This information is further supported by [Document 2], [Document 3], and [Document 5].\n\nExample 2:\n\n"
    #         question = qasper_train_dict[each][0]
    #         answer_info = qasper_train_dict[each][1]
    #         evidence = qasper_train_dict[each][2]
    #         full_text = qasper_train_dict[each][5]
    #         if answer_info["unanswerable"]:
    #             answer = "Unanswerable"
    #         else:
    #             if answer_info["extractive_spans"]:
    #                 answer = ", ".join(answer_info["extractive_spans"])
    #                 answer_type = "extractive"
    #             elif answer_info["free_form_answer"]:
    #                 answer = answer_info["free_form_answer"]
    #                 answer_type = "abstractive"
    #             elif answer_info["yes_no"]:
    #                 answer = "Yes"
    #                 answer_type = "boolean"
    #             elif answer_info["yes_no"] is not None:
    #                 answer = "No"
    #                 answer_type = "boolean"
    #
    #         evidence = [text for text in evidence if "FLOAT SELECTED" not in text]
    #
    #         # pertubed_evidence = evidence + random_pertube(full_text, evidence)
    #         #
    #         # random.shuffle(pertubed_evidence)
    #         # use only evidence
    #         for i, d in enumerate(evidence):
    #             prompt += "[Document {0}]: {1}\n\n".format(i + 1, d)
    #         prompt += "Question: {0}\n\nAnswer: {1}\n\nExplanation:".format(question, answer)
    #         res = generate(prompt)
    #         explained_questions.append({
    #             "question": question,
    #             "evidence": evidence,
    #             "answer": answer,
    #             "explanation": res
    #         })
    #     return explained_questions

    for k in data:
        for qa in data[k]['qas']:
            questions[qa['question_id']] = qa
            abstract_dict[qa['question_id']] = data[k]['abstract']
            title_dict[qa['question_id']] = data[k]['title']
            intro_dict[qa['question_id']] = data[k]['full_text'][0]['paragraphs']

    def decompose(question):
        prompt = "Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.\n\nExample 1:\n\nQuestion: Is Hamlet more common on IMDB than Comedy of Errors?\n\nDecompositions: \n1: How many listings of Hamlet are there on IMDB?\n2: How many listing of Comedy of Errors is there on IMDB?\n\nExample 2:\n\nQuestion: Are birds important to badminton?\n\nDecompositions:\nThe question needs no decomposition\n\nExample 3:\n\nQuestion: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?\n\nDecompositions:\n1: What is the minimum driving age in the US?\n2: What is the minimum age for someone to be employed in the US?\n\nExample 4:\n\nQuestion: Are all cucumbers the same texture?\n\nDecompositions:\nThe question needs no decomposition\n\nExample 5:\n\nQuestion: Hydrogen's atomic number squared exceeds number of Spice Girls?\n\nDecompositions:\n1: What is the atomic number of hydrogen?\n2: How many Spice Girls are there?\n\nExample 6:\n\nQuestion: {0}\n\nDecompositions:"

        res = generate(prompt.format(question), max_tokens=256)
        # print(res)
        if res.lower().strip() == "the question needs no decomposition.":
            return [question]
        try:
            questions = [l for l in res.splitlines() if l != ""]
            questions = [q.split(':')[1].strip() for q in questions]
            return questions
        except:
            return [question]

    extractives = []
    abstractives = []
    none_ = []
    yes_no = []
    stratfy_by = []
    for item in qas:
        annotation_info = questions[item['question_id']]["answers"][0]
        # annotation_info = item['answers'][0]
        answer_info = annotation_info["answer"]
        if answer_info["unanswerable"]:
            stratfy_by.append(0)
            none_.append(item)
        else:
            if answer_info["extractive_spans"]:
                extractives.append(item)
                stratfy_by.append(1)
            elif answer_info["free_form_answer"]:
                # answer = answer_info["free_form_answer"]
                abstractives.append(item)
                stratfy_by.append(2)
                # answer_type = "abstractive"
            elif answer_info["yes_no"]:
                yes_no.append(item)
                stratfy_by.append(3)
            elif answer_info["yes_no"] is not None:
                yes_no.append(item)
                stratfy_by.append(3)
    print(len(extractives))
    print(len(abstractives))
    print(len(none_))
    print(len(yes_no))

    # 732
    # 371
    # 139
    # 209
    # 218

    from sklearn.model_selection import train_test_split
    X_train, test, y_train, y_test = train_test_split(qas, stratfy_by, test_size=0.15, random_state=42,
                                                      stratify=stratfy_by)
    print(len(test))

    # from sentence_transformers import SentenceTransformer, util
    # model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

    # train = json.load(open("explained_train_1_5.json",'r'))
    # train = json.load(open("../explained_train_ori_evidence_5_50.json",'r'))
    # train_embeddings = model.encode([a['question'] for a in train])

    import os
    import openai

    # from transformers import GPT2Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    openai.api_key = ('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')

    def generate(prompt, max_tokens=1000, temperature=0.0):
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
            break

        return response["choices"][0]['text']

    def chatgpt_generate(prompt, max_tokens=4096, temperature=0.0):
        while True:
            # try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(response)
            time.sleep(1)
            break
            # except Exception as e:
            #     print(Exception)
            #     time.sleep(5)
            #     continue
            # break

        return response['choices'][0]['message']['content']

    import random

    random.seed(101)
    kshot = 3
    RANDOM = True
    STATIC = True

    def generate_explanation_answer(question, documents, evidence, abstract, entity_desc, explained_knn_questions):
        # zero-shot no context qa
        # prompt = "For each example, create an \"Answer\" and to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"
        # intructions of using the context,
        prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"

        for i, hit in enumerate(explained_knn_questions):
            prompt += "Example {0}:\n\n".format(i + 1)
            for j, d in enumerate(hit['evidence']):
                prompt += "[Document {0}]: {1}\n\n".format(j + 1, d)
            prompt += "Question: {0}\n\nExplanation: {1}\n\nAnswer: {2}\n\n".format(hit['question'], hit['explanation'],
                                                                                    hit['answer'])

        # This prompt has 4 examples, each with a question, explanation, and answer including unanswerable .

        prompt += "Example {0}:\n\n".format(4)
        #
        # for k, document in enumerate(documents):
        #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document['title'])

        # for k, document in enumerate(evidence):
        #    prompt += "[Document {0}]: {1}\n\n".format(k + 1, document)

        # for k, document in enumerate(entity_desc):
        #     prompt += "[Entity {0}]: {1}\n\n".format(k + 1, document)

        prompt += "Question: {0}\n\nExplanation:".format(question)

        tokens = tokenizer.tokenize(prompt)
        # print(prompt)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).input_ids.to("cuda")
        outputs = model.generate(input_ids, temperature=0, top_k=25, top_p=1, no_repeat_ngram_size=10,
                                 early_stopping=True)
        res = tokenizer.decode(outputs[0])

        if "answer:" in res.lower():
            answer = res.lower().split("answer:")[1]
        else:
            answer = res
        return answer, answer, prompt, res

    import re
    from tqdm import tqdm
    import numpy as np
    from datetime import date
    import random

    RANDOM = True

    kshot = 3
    docs = 3

    knn_dict = {}
    if __name__ == '__main__':
        regex = r"\[document \d+\]"
        f1 = open("qasper_flant5xl_top_p1_knn_nocontext_random101.jsonl", 'w')
        f_log = open("qasper_flant5xl_top_p1_knn_nocontext_random101_log.jsonl", 'w')
        logs = []
        tokens = []
        num = 0
        qasper_knn_test_dict = json.load(open("qasper_knn_test_result.json"))
        # q_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_q_entity_desc_test.json"))
        # evi_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_evi_entity_desc_test.json"))
        knn_result = {}
        batch_size = 20
        for i in range(0, len(test), batch_size):
            batch_item = test[i: i + batch_size]
            documents = batch_item['documents']
            evidence = []
            abstract = abstract_dict[batch_item['question_id']]
            annotation_info = questions[batch_item['question_id']]["answers"]
            # annotation_info = item['answers'][0]\
            for each in annotation_info:
                answer_info = each["answer"]
                evidence.extend(answer_info['evidence'])

            evidence_0 = questions[batch_item['question_id']]["answers"][0]["answer"]['evidence']
            # q_entity = q_entity_dict.get(item['question_id'],"")
            # evi_entity = evi_entity_dict.get(item['question_id'],"")

            entity_desc = []
            # entity_desc.append(q_entity)
            # entity_desc.append(evi_entity)

            explained_knn_questions = qasper_knn_test_dict[batch_item['question_id']]

            # knn_result[item['question_id']] = explained_knn_questions

            # json.dump(knn_result, open("qasper_knn_test_result.json", 'w'))
            explanation1, answer1, prompt1, res1 = generate_explanation_answer(batch_item['question'], documents[:docs],
                                                                               evidence_0, abstract, entity_desc,
                                                                               explained_knn_questions)
            # explanation, answer, prompt, res = generate_explanation_answer(item['question'], documents[:docs],evidence_0,abstract, entity_desc,explained_knn_questions)

            matches = re.finditer(regex, explanation1, re.MULTILINE)
            relevant_documents = []
            for match in matches:
                nums = re.findall(r'\b\d+\b', match.group())
                if len(nums) > 0:
                    try:
                        relevant_documents.append(documents[int(nums[0]) - 1]['title'])
                    except:
                        pass

            f1.write(json.dumps({
                'question_id': batch_item['question_id'],
                'question': batch_item['question'],
                'predicted_answer': answer1.rstrip(),
                'predicted_evidence': evidence_0,
                'explanation': explanation1
            }) + "\n")

            f_log.write(json.dumps({
                'question_id': batch_item['question_id'],
                "prompt": prompt1
            }) + "\n")

    # This prompt has 4 examples, each with a question, explanation, and answer including unanswerable .



    prompt += "Example {0}:\n\n".format(4)
    #
    # for k, document in enumerate(documents):
    #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document['title'])

    #for k, document in enumerate(evidence):
    #    prompt += "[Document {0}]: {1}\n\n".format(k + 1, document)

    # for k, document in enumerate(entity_desc):
    #     prompt += "[Entity {0}]: {1}\n\n".format(k + 1, document)

    prompt += "Question: {0}\n\nExplanation:".format(question)

    tokens = tokenizer.tokenize(prompt)
    # print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).input_ids.to("cuda")
    outputs = model.generate(input_ids, temperature=0, top_k=25, top_p=1,no_repeat_ngram_size=10, early_stopping=True)
    res = tokenizer.decode(outputs[0])

    if "answer:" in res.lower():
        answer = res.lower().split("answer:")[1]
    else:
        answer = res
    return answer, answer, prompt, res


import re
from tqdm import tqdm
import numpy as np
from datetime import date
import random

RANDOM = True

kshot = 3
docs = 3

knn_dict = {}
if __name__ == '__main__':
    regex = r"\[document \d+\]"
    f1 = open("qasper_flant5xl_top_p1_knn_nocontext_random101.jsonl", 'w')
    f_log = open("qasper_flant5xl_top_p1_knn_nocontext_random101_log.jsonl", 'w')
    logs = []
    tokens= []
    num = 0
    qasper_knn_test_dict = json.load(open("qasper_knn_test_result.json"))
    # q_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_q_entity_desc_test.json"))
    # evi_entity_dict = json.load(open("/Users/wenbingbing/PycharmProjects/galactica/qasper_evi_entity_desc_test.json"))
    knn_result = {}
    batch_size = 20
    for i in range(0, len(test), batch_size):
        batch_item = test[i: i + batch_size]
        documents = batch_item['documents']
        evidence = []
        abstract = abstract_dict[batch_item['question_id']]
        annotation_info = questions[batch_item['question_id']]["answers"]
        # annotation_info = item['answers'][0]\
        for each in annotation_info:
            answer_info = each["answer"]
            evidence.extend(answer_info['evidence'])

        evidence_0 = questions[batch_item['question_id']]["answers"][0]["answer"]['evidence']
        # q_entity = q_entity_dict.get(item['question_id'],"")
        # evi_entity = evi_entity_dict.get(item['question_id'],"")

        entity_desc = []
        # entity_desc.append(q_entity)
        # entity_desc.append(evi_entity)

        explained_knn_questions = qasper_knn_test_dict[batch_item['question_id']]

        # knn_result[item['question_id']] = explained_knn_questions

    # json.dump(knn_result, open("qasper_knn_test_result.json", 'w'))
        explanation1, answer1, prompt1, res1 = generate_explanation_answer(batch_item['question'], documents[:docs],evidence_0, abstract, entity_desc, explained_knn_questions)
        # explanation, answer, prompt, res = generate_explanation_answer(item['question'], documents[:docs],evidence_0,abstract, entity_desc,explained_knn_questions)

        matches = re.finditer(regex, explanation1, re.MULTILINE)
        relevant_documents = []
        for match in matches:
            nums = re.findall(r'\b\d+\b', match.group())
            if len(nums) > 0:
                try:
                    relevant_documents.append(documents[int(nums[0])-1]['title'])
                except:
                    pass

        f1.write(json.dumps({
            'question_id': batch_item['question_id'],
            'question': batch_item['question'],
            'predicted_answer': answer1.rstrip(),
            'predicted_evidence': evidence_0,
            'explanation': explanation1
        })+"\n")

        f_log.write(json.dumps({
            'question_id': batch_item['question_id'],
            "prompt": prompt1
        })+"\n")

