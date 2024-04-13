echo "no context sota llama freeform"
python pubmed_evaluation.py  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
#500
#19.0 88.0 87.0
#all 0.388000
#noans_acc0.345455
#yes_acc0.615942
#no_acc0.396450
#noans_yes0.318841
#noans_no0.514793
#all but noans 0.393258
#Accuracy 0.512000
#Macro-F1 0.452620
#[[170  18  88]
# [ 15  67  87]
# [ 25  11  19]]
#context_length 202.272
#500
#13.0 53.0 29.0
#all 0.190000
#noans_acc0.236364
#yes_acc0.246377
#no_acc0.704142
#noans_yes0.192029
#noans_no0.171598
#all but noans 0.184270
#Accuracy 0.400000
#Macro-F1 0.345786
#[[ 68 155  53]
# [ 21 119  29]
# [ 13  29  13]]
#context_length 202.272
#hashas 0.524
#hasno 0.088
#nohas 0.286
#nono 0.102

echo "no context sota vicuna"
python pubmed_evaluation.py  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \

#500
#33.0 129.0 149.0
#all 0.622000
#noans_acc0.600000
#yes_acc0.532609
#no_acc0.000000
#noans_yes0.467391
#noans_no0.881657
#all but noans 0.624719
#Accuracy 0.360000
#Macro-F1 0.270862
#[[147   0 129]
# [ 20   0 149]
# [ 22   0  33]]
#context_length 202.272
#500
#48.0 239.0 154.0
#all 0.882000
#noans_acc0.872727
#yes_acc0.134058
#no_acc0.000000
#noans_yes0.865942
#noans_no0.911243
#all but noans 0.883146
#Accuracy 0.170000
#Macro-F1 0.138148
#[[ 37   0 239]
# [ 15   0 154]
# [  7   0  48]]
#context_length 202.272
#hashas 0.086
#hasno 0.292
#nohas 0.032
#nono 0.59


echo "no context sota flan"

python3 pubmed_evaluation.py  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \



echo "no context sota gpt"
python3 pubmed_evaluation.py  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \

#500
#15.0 34.0 83.0
#all 0.264000
#noans_acc0.272727
#yes_acc0.851449
#no_acc0.260355
#noans_yes0.123188
#noans_no0.491124
#all but noans 0.262921
#Accuracy 0.588000
#Macro-F1 0.450163
#[[235   7  34]
# [ 42  44  83]
# [ 34   6  15]]
#context_length 202.272
#500
#33.0 183.0 127.0
#all 0.686000
#noans_acc0.600000
#yes_acc0.322464
#no_acc0.011834
#noans_yes0.663043
#noans_no0.751479
#all but noans 0.696629
#Accuracy 0.248000
#Macro-F1 0.201849
#[[ 89   4 183]
# [ 40   2 127]
# [ 22   0  33]]
#context_length 202.272
#hashas 0.274
#hasno 0.462
#nohas 0.04
#nono 0.224



echo "random context sota llama"
python3 pubmed_evaluation.py  \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl \
##140
#all 0.000000
#Accuracy 0.985714
#Macro-F1 0.983811
#140
#all 0.000000
#Accuracy 0.357143
#Macro-F1 0.293564
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0

echo "random context sota vicuna"
python3 pubmed_evaluation.py \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

#140
#all 0.000000
#Accuracy 0.942857
#Macro-F1 0.931973
#139
#all 0.000000
#Accuracy 0.575540
#Macro-F1 0.571190
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0

echo "random context sota flan"
python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##



echo "random context sota gpt"

python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##




echo "double random context sota llama"
 python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

#140
#all 0.000000
#Accuracy 0.985714
#Macro-F1 0.983811
#140
#all 0.000000
#Accuracy 0.950000
#Macro-F1 0.943649
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0

echo "double random context sota vicuna"
 python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

#140
#all 0.000000
#Accuracy 0.942857
#Macro-F1 0.931973
#140
#all 0.000000
#Accuracy 0.928571
#Macro-F1 0.913772
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0
echo "double random context sota flan"

 python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl





echo "double random context sota gpt"
 python3 pubmed_evaluation.py \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl




## no context sota









##random context sota


## doublerandom context sota

#python pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
##500
#19.0 88.0 87.0
#all 0.388000
#noans_acc0.345455
#yes_acc0.615942
#no_acc0.396450
#noans_yes0.318841
#noans_no0.514793
#all but noans 0.393258
#Accuracy 0.512000
#Macro-F1 0.452620
#[[170  18  88]
# [ 15  67  87]
# [ 25  11  19]]
#context_length 202.272
#500
#35.0 118.0 126.0
#all 0.558000
#noans_acc0.636364
#yes_acc0.539855
#no_acc0.147929
#noans_yes0.427536
#noans_no0.745562
#all but noans 0.548315
#Accuracy 0.418000
#Macro-F1 0.366873
#[[149   9 118]
# [ 18  25 126]
# [ 15   5  35]]
#context_length 202.272
#hashas 0.402
#hasno 0.21
#nohas 0.04
#nono 0.348





## no context freeform

#python pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \

#500
#53.0 268.0 163.0
#all 0.968000
#noans_acc0.963636
#yes_acc0.000000
#no_acc0.000000
#noans_yes0.971014
#noans_no0.964497
#all but noans 0.968539
#Accuracy 0.106000
#Macro-F1 0.010351
#[[  0   0 268]
# [  0   0 163]
# [  0   0  53]]
#context_length 202.272
#500
#29.0 136.0 88.0
#all 0.506000
#noans_acc0.527273
#yes_acc0.086957
#no_acc0.005917
#noans_yes0.492754
#noans_no0.520710
#all but noans 0.503371
#Accuracy 0.108000
#Macro-F1 0.001653
#[[ 24   0 136]
# [  7   1  88]
# [  2   1  29]]
#context_length 202.272
#hashas 0.018
#hasno 0.014
#nohas 0.476
#nono 0.492


#python pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
##500
#45.0 221.0 163.0
#all 0.858000
#noans_acc0.818182
#yes_acc0.199275
#no_acc0.005917
#noans_yes0.800725
#noans_no0.964497
#all but noans 0.862921
#Accuracy 0.202000
#Macro-F1 0.171878
#[[ 55   0 221]
# [  5   1 163]
# [ 10   0  45]]
#context_length 202.272
#500
#46.0 224.0 147.0
#all 0.834000
#noans_acc0.836364
#yes_acc0.173913
#no_acc0.011834
#noans_yes0.811594
#noans_no0.869822
#all but noans 0.833708
#Accuracy 0.192000
#Macro-F1 0.061622
#[[ 48   2 224]
# [ 18   2 147]
# [  7   1  46]]
#context_length 202.272
#hashas 0.042
#hasno 0.1
#nohas 0.124
#nono 0.734





##random context freeform


## doublerandom context freeform

#python pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
#

#python pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/pubmedqa/pubmedqa_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \







# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/llamav2_0shot_context_exp2_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/llamav2_0shot_nocontext_exp2_doc_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/llamav2_0shot_nocontext_exp2_doc_comp_pubmed.jsonl


# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/pubmedllamav2_0shot_context_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/pubmedllamav2_0shot_nocontext_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/pubmedllamav2_0shot_nocontext_comp_pubmed.jsonl


# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/pubmedllamav2_0shot_context_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/llamav2_0shot_randomcontext_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/llamav2_0shot_randomcontext_comp_pubmed.jsonl



# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_context_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_nocontext_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_nocontext_comp_pubmed.jsonl


# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_context_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_randomcontext_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/vicuna1.5_13b_0shot_randomcontext_comp_pubmed.jsonl


# python pubmed_evaluation.py  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_context_pubmed.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_nocontext_pubmed.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_nocontext_comp_pubmed.jsonl


#python pubmed_evaluation.py  \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_context_pubmed.jsonl  \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_randomcontext_pubmed.jsonl \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/pubmed/flanxl_0shot_randomcontext_comp_pubmed.jsonl

#
#

#python3 pubmed_evaluation.py  \
#/Users/wenbingbing/PycharmProjects/qasper/results/pubmed/pubmed_chatgpt_0shot_context_new.jsonl  \
#/Users/wenbingbing/PycharmProjects/qasper/results/pubmed/chatgpt_0shot_context_prompt1_exp2_pubmed.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/pubmed/chatgpt_0shot_context_prompt1_exp2_pubmed_comp.jsonl
#all 0.452000
#noans_acc0.555556
#yes_acc0.623188
#no_acc0.305882
#noans_yes0.333333
#noans_no0.611765
#all but noans 0.439462
#Accuracy 0.508000
#Macro-F1 0.454984
#[[86  6 46]
# [ 7 26 52]
# [10  2 15]]
#context_length 202.272
#10.0 18.0 27.0
#all 0.220000
#noans_acc0.370370
#yes_acc0.362319
#no_acc0.247059
#noans_yes0.130435
#noans_no0.317647
#all but noans 0.201794
#Accuracy 0.547297
#Macro-F1 0.496496
#[[50 11 18]
# [ 5 21 27]
# [ 5  1 10]]
#context_length 202.272
#hashas 0.4594594594594595
#hasno 0.0945945945945946
#nohas 0.16891891891891891
#nono 0.27702702702702703

