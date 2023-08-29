import json
import time
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

# if STATIC:
#     static_sample = random.sample(train, k=kshot)
#     # print(static_sample)


def generate_explanation_answer(question, context):
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\n"
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
    # for i, hit in enumerate(hits):
    #     prompt += "Example {0}:\n\n".format(i + 1)
    #
    #     for j, d in enumerate(hit['evidence']):
    #         prompt += "[Document {0}]: {1}\n\n".format(j + 1, d)
    #
    #     prompt += "Question: {0}\n\nExplanation: {1}\n\nAnswer: {2}\n\n".format(hit['question'], hit['explanation'],
    #                                                                             hit['answer'])



    # This prompt has 4 examples, each with a question, explanation, and answer including unanswerable .
    # prompt = "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"yes\" or \"no\" in boolean questions.\n\nExample 1:\n\n[Document 1]: In Figure FIGREF8 , we see that overall sentiment averages rarely show movement post-event: that is, only Hurricane Florence shows a significant difference in average tweet sentiment pre- and post-event at the 1% level, corresponding to a 0.12 point decrease in positive climate change sentiment. However, controlling for the same group of users tells a different story: both Hurricane Florence and Hurricane Michael have significant tweet sentiment average differences pre- and post-event at the 1% level. Within-cohort, Hurricane Florence sees an increase in positive climate change sentiment by 0.21 points, which is contrary to the overall average change (the latter being likely biased since an influx of climate change deniers are likely to tweet about hurricanes only after the event). Hurricane Michael sees an increase in average tweet sentiment of 0.11 points, which reverses the direction of tweets from mostly negative pre-event to mostly positive post-event. Likely due to similar bias reasons, the Mendocino wildfires in California see a 0.06 point decrease in overall sentiment post-event, but a 0.09 point increase in within-cohort sentiment. Methodologically, we assert that overall averages are not robust results to use in sentiment analyses.\n\n[Document 2]: The second data batch consists of event-related tweets for five natural disasters occurring in the U.S. in 2018. These are: the East Coast Bomb Cyclone (Jan. 2 - 6); the Mendocino, California wildfires (Jul. 27 - Sept. 18); Hurricane Florence (Aug. 31 - Sept. 19); Hurricane Michael (Oct. 7 - 16); and the California Camp Fires (Nov. 8 - 25). For each disaster, we scraped tweets starting from two weeks prior to the beginning of the event, and continuing through two weeks after the end of the event. Summary statistics on the downloaded event-specific tweets are provided in Table TABREF1 . Note that the number of tweets occurring prior to the two 2018 sets of California fires are relatively small. This is because the magnitudes of these wildfires were relatively unpredictable, whereas blizzards and hurricanes are often forecast weeks in advance alongside public warnings. The first (influential tweet data) and second (event-related tweet data) batches are de-duplicated to be mutually exclusive. In Section SECREF2 , we perform geographic analysis on the event-related tweets from which we can scrape self-reported user city from Twitter user profile header cards; overall this includes 840 pre-event and 5,984 post-event tweets.\n\nQuestion: Which five natural disasters were examined?\n\nExplanation:  The five natural disasters examined in [Document 2] are the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires. This information is further supported by [Document 1].\n\nAnswer: the East Coast Bomb Cyclone,  the Mendocino, California wildfires, Hurricane Florence, Hurricane Michael, the California Camp Fires\n\nExample 2:\n\n[Document 1]: N-GrAM ranked first in all cases except for the language variety task. In this case, the baseline was the top-ranked system, and ours was second by a small margin. Our system significantly out-performed the baseline on the joint task, as the baseline scored significantly lower for the gender task than for the variety task.\n\n[Document 2]: ()\n\nQuestion: On which task does do model do worst?\n\nExplanation: \n\nAccording to [Document 1], the model does worst on the gender prediction task.\n\nAnswer: Gender prediction task\n\nExample 3:\n\n[Document 1]: As seen in Table TABREF38, both SMERTI variations achieve higher STES and outperform the other models overall, with the WordNet models performing the worst. SMERTI excels especially on fluency and content similarity. The transformer variation achieves slightly higher SLOR, while the RNN variation achieves slightly higher CSS. The WordNet models perform strongest in sentiment preservation (SPA), likely because they modify little of the text and only verbs and nouns. They achieve by far the lowest CSS, likely in part due to this limited text replacement. They also do not account for context, and many words (e.g. proper nouns) do not exist in WordNet. Overall, the WordNet models are not very effective at STE.\n\n[Document 2]: We evaluate on three datasets: Yelp and Amazon reviews BIBREF1, and Kaggle news headlines BIBREF2. We implement three baseline models for comparison: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), and Word2Vec Semantic Text Exchange Model (W2V-STEM).\n\nQuestion: What are the baseline models mentioned in the paper?\n\nExplanation:  The baseline models are NWN-STEM, GWN-STEM, and W2V-STEM. This is mentioned in [Document 2].\n\nAnswer: Noun WordNet Semantic Text Exchange Model (NWN-STEM), General WordNet Semantic Text Exchange Model (GWN-STEM), Word2Vec Semantic Text Exchange Model (W2V-STEM)\n\nExample 4:\n\n[Document 1]: ()\n\nQuestion: Are there privacy concerns with clinical data?\n\nExplanation: \n\nThe documents don't give enough information to answer the question.\n\nAnswer: unanswerable\n\nExample 5:\n\n"


    # This prompt has 3 examples, each with a question, explanation, and answer.
    ##2shots and no context
    prompt = "For each example, create an concise \"Answer\" to the \"Question\". Pay attention to answer only \"yes\" or \"no\".\n\nExample 1:\n\nQuestion: Is the protein Papilin secreted?\n\nAnswer: yes.\n\nExample 2:\n\nQuestion: Is polyadenylation a process that stabilizes a protein by adding a string of Adenosine residues to the end of the molecule?\n\nAnswer: no.\n\n"
    ##2shots and context
    # prompt = "For each example, use the documents to create an concise \"Answer\" to the \"Question\". Pay attention to answer only \"yes\" or \"no\".\n\nExample 1:\n\n[Document 1]:Using expression analysis, we identify three genes that are transcriptionally regulated by HLH-2: the protocadherin cdh-3, and two genes encoding secreted extracellular matrix proteins, mig-6/papilin and him-4/hemicentin.  We found that mig-6 encodes long (MIG-6L) and short (MIG-6S) isoforms of the extracellular matrix protein papilin, each required for distinct aspects of DTC migration. Both MIG-6 isoforms have a predicted N-terminal papilin cassette apilins are homologous, secreted extracellular matrix proteins which share a common order of protein domains.  The TSR superfamily is a diverse family of extracellular matrix and transmembrane proteins, many of which have functions related to regulating matrix organization, cell-cell interactions and cell guidance. This review samples some of the contemporary literature regarding TSR superfamily members (e.g. F-spondin, UNC-5, ADAMTS, papilin, and TRAP) where specific functions are assigned to the TSR domains. Papilins are extracellular matrix proteins  Papilin is an extracellular matrix glycoprotein   Collagen IV, laminin, glutactin, papilin, and other extracellular matrix proteins were made primarily by hemocytes and were secreted into the medium.  A sulfated glycoprotein was isolated from the culture media of Drosophila Kc cells and named papilin.\n\nQuestion: Is the protein Papilin secreted?\n\nAnswer: yes.\n\nExample 2:\n\n[Document 1]:The addition of poly(A) tails to eukaryotic nuclear mRNAs promotes their stability, export to the cytoplasm and translation.  Most eukaryotic genes express mRNAs with alternative polyadenylation sites at their 3' ends Polyadenylation is the non-template addition of adenosine nucleotides at the 3'-end of RNA, which occurs after transcription and generates a poly(A) tail up to 250-300 nucleotides long. Polyadenylation is a process of endonucleolytic cleavage of the mRNA, followed by addition of up to 250 adenosine residues to the 3' end of the mRNA. Plant mitochondrial polyadenylated mRNAs are degraded by a 3'- to 5'-exoribonuclease activity, which proceeds unimpeded by stable secondary structures. We show that a 3'- to 5'-exoribonuclease activity is responsible for the preferential degradation of polyadenylated mRNAs as compared with non-polyadenylated mRNAs, and that 20-30 adenosine residues constitute the optimal poly(A) tail size for inducing degradation of RNA substrates in vitro. The diversity of polyadenylation sites suggests that mRNA polyadenylation in prokaryotes is a relatively indiscriminate process that can occur at all mRNA's 3'-ends and does not require specific consensus sequences as in eukaryotes. Polyadenylation of premessenger RNAs occurs posttranscriptionally in the nucleus of eukaryotic cells by cleavage of the precursor and polymerization of adenosine residues. However, under certain conditions, poly(A) tracts may lead to mRNA stabilization. From these results, we propose that in plant mitochondria, poly(A) tails added at the 3' ends of mRNAs promote an efficient 3'- to 5'- degradation process.. Auxiliary downstream elements are required for efficient polyadenylation of mammalian pre-mRNAs. Transcription in these cells is polycistronic. Tens to hundreds of protein-coding genes of unrelated function are arrayed in long clusters on the same DNA strand. Polycistrons are cotranscriptionally processed by trans-splicing at the 5' end and polyadenylation at the 3' end, generating monocistronic units ready for degradation or translation We have devised a simple chromatographic procedure which isolates five polyadenylation factors that are required for polyadenylation of eukaryotic mRNA.  During mammalian oocyte maturation, protein synthesis is mainly controlled through cytoplasmic polyadenylation of stored maternal mRNAs. Identification and characterization of a polyadenylated small RNA (s-poly A+ RNA) in dinoflagellates. Thus, polyadenylation seems to be a major component of the RNA editing machinery that affects overlapping genes in animal mitochondria. Pre-mRNA 3'-end processing, the process through which almost all eukaryotic mRNAs acquire a poly(A) tail is generally inhibited during the cellular DNA damage Almost all eukaryotic mRNAs possess 3' ends with a polyadenylate (poly(A)) tail. We previously demonstrated, by limited mutagenesis, that conserved sequence elements within the 5' end of influenza virus virion RNA (vRNA) are required for the polyadenylation of mRNA in vitro. Polyadenylation of mRNA precursors by poly(A) polymerase depends on two specificity factors and their recognition sequences The majority of eukaryotic pre-mRNAs are processed by 3'-end cleavage and polyadenylation Formation of mRNA 3' termini involves cleavage of an mRNA precursor and polyadenylation of the newly formed end.  The polyadenylation of RNA is a near-universal feature of RNA metabolism in eukaryotes. The mechanism of RNA degradation in Escherichia coli involves endonucleolytic cleavage, polyadenylation of the cleavage product by poly(A) polymerase, and exonucleolytic degradation by the exoribonucleases,  The addition of poly(A)-tails to RNA is a process common to almost all organisms.  The addition of poly(A) tails to RNA is a phenomenon common to all organisms examined so far.  The addition of poly(A)-tails to RNA is a phenomenon common to almost all organisms.  Polyadenylation contributes to the destabilization of bacterial mRNA.\n\nQuestion: Is polyadenylation a process that stabilizes a protein by adding a string of Adenosine residues to the end of the molecule?\n\nAnswer: no.\n\n"

    prompt += "Example {0}:\n\n".format(3)

    # for k, document in enumerate(documents):
    #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document['title'])

    # for k, document in enumerate(context):
    #     prompt += "[Document {0}]: {1}\n\n".format(k + 1, document)

    # for k, document in enumerate(entity_desc):
    #     prompt += "[Entity {0}]: {1}\n\n".format(k + 1, document)

    prompt += "Question: {0}\n\n".format(question)

    tokens = tokenizer.tokenize(prompt)
    # print(prompt)
    res = chatgpt_generate(prompt)

    # if "answer:" not in res.lower():
    #     prompt = prompt + res + "\n\nAnswer:"
    #     res = res + "\n\nAnswer:" + generate(prompt)

    # explanation = res.lower().split("answer:")[0]
    # answer = res.lower().split("answer:")[1]

    return  res


import re
from tqdm import tqdm
import numpy as np
from datetime import date
import random

RANDOM = True

kshot = 3
docs = 3

regex = r"\[document \d+\]"

f = open("chatgpt_2_shot_biqasq_test.jsonl", "w")
data = open("bioasq_hf/test.json", "r")
background = {}
for line in open("gpt_background_biqasq_test.jsonl"):
    each = json.loads(line)
    background[each["question_id"]] = each["background"].split("\n\n")[-1]
stratfy_by = []
all = []
for line in data:
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

# X_train, test = train_test_split(all, test_size=0.5, random_state=42, stratify=stratfy_by)
# print(len(test))

for each in tqdm(all):
    question_id = each['id']
    question = each["sentence1"]
    context = [each["sentence2"]]
    answer = each["label"]
    # context = [background[question_id]]
    res = generate_explanation_answer(question, context)

    f.write(json.dumps({
        'question_id': question_id,
        'question':question,
        'predicted_answer': res,
        'explanation': res
    })+"\n")

    # f_log.write(json.dumps({
    #     'question_id': item['question_id'],
    #     "prompt": prompt
    # })+"\n")
