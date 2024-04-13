 echo "no context sota llama"
 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
#
#140
#all 0.000000
#Accuracy 0.985714
#Macro-F1 0.983811
#140
#all 0.000000
#Accuracy 0.557143
#Macro-F1 0.556781
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0
#
echo "no context sota vicuna"
python3 bioasq_evaluation.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
##
#140
#all 0.000000
#Accuracy 0.942857
#Macro-F1 0.931973
#140
#all 0.107143
#Accuracy 0.607143
#Macro-F1 0.273798
#hashas 0.8928571428571429
#hasno 0.10714285714285714
#nohas 0.0
#nono 0.0


echo "no context sota flan"
python3 bioasq_evaluation.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \


#no context sota flan
#140
#all 0.000000
#Accuracy 0.971429
#Macro-F1 0.967971
#140
#all 0.000000
#Accuracy 0.657143
#Macro-F1 0.559171
#same 0.6428571428571429
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0
#


echo "no context sota gpt"
 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
##
#
#140
#all 0.000000
#Accuracy 0.978571
#Macro-F1 0.975580
#140
#all 0.357143
#Accuracy 0.514286
#Macro-F1 0.353559
#hashas 0.6428571428571429
#hasno 0.35714285714285715
#nohas 0.0
#nono 0.0



echo "random context sota llama"
python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 bioasq_evaluation.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##



echo "random context sota gpt"

python3 bioasq_evaluation.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##




echo "double random context sota llama"
 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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
 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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

 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl





echo "double random context sota gpt"
 python3 bioasq_evaluation.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq/bioasq_gpt-3.5-turbo_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

#140
#all 0.000000
#Accuracy 0.978571
#Macro-F1 0.975580
#140
#all 0.000000
#Accuracy 0.935714
#Macro-F1 0.927549
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0







 #no context freeform
# python3 bioasq_evaluation.py \
# --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
# --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
# --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl

#140
#all 0.000000
#Accuracy 0.800000
#Macro-F1 0.551682
#140
#all 0.557143
#Accuracy 0.100000
#Macro-F1 0.074277
#hashas 0.44285714285714284
#hasno 0.5571428571428572
#nohas 0.0
#nono 0.0
#
#
#

#python3 bioasq_evaluation.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl

#140
#all 0.000000
#Accuracy 0.757143
#Macro-F1 0.049901
#140
#all 0.542857
#Accuracy 0.342857
#Macro-F1 0.203822
#hashas 0.45714285714285713
#hasno 0.5428571428571428
#nohas 0.0
#nono 0.0


# python3 bioasq_evaluation.py \
# --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_gpt-3.5-turbo_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
# --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_gpt-3.5-turbo_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
# --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_gpt-3.5-turbo_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
###
#130
#all 0.000000
#Accuracy 0.976923
#Macro-F1 0.971892
#140
#all 0.292857
#Accuracy 0.585714
#Macro-F1 0.307010
#hashas 0.7153846153846154
#hasno 0.2846153846153846
#nohas 0.0
#nono 0.0


# random context freeform
#python3 bioasq_evaluation.py \
# --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
# --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
# --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

#133
#all 0.000000
#Accuracy 0.962406
#Macro-F1 0.957298
#126
#all 0.000000
#Accuracy 0.420635
#Macro-F1 0.400665
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0





 #double random context freeform
# python3 bioasq_evaluation.py \
# --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
# --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
# --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/bioasq_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
##
#133
#all 0.000000
#Accuracy 0.962406
#Macro-F1 0.957298
#131
#all 0.000000
#Accuracy 0.946565
#Macro-F1 0.939770
#hashas 1.0
#hasno 0.0
#nohas 0.0
#nono 0.0









# python bioasq_evaluation.py \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/vicuna1.5_13b_0shot_context_bioasq.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/vicuna1.5_13b_0shot_randomcontext_bioasq.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/vicuna1.5_13b_0shot_randomcontext_comp_bioasq.jsonl


# python bioasq_evaluation.py \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/llamav2_13b_0shot_context_bioasq.jsonl  \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/llamav2_13b_0shot_0context_bioasq.jsonl \
# /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/llamav2_13b_0shot_0context_comp_bioasq.jsonl

#llama 0context could update unanswerable


#python bioasq_evaluation.py \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/flanxl_0shot_context_bioasq.jsonl  \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/flanxl_0shot_randomcontext_bioasq.jsonl \
#/home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/bioasq/flanxl_0shot_randomcontext_comp_bioasq.jsonl

#python3 bioasq_evaluation.py \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_biqasq_context.jsonl  \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_context_prompt1_exp2_bioasq.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_context_prompt1_exp2_bioasq_comp.jsonl
#all 0.114286
#Accuracy 0.842857
#Macro-F1 0.555985
#all 0.000000
#Accuracy 0.971429
#Macro-F1 0.968297
#retain_has 0.8214285714285714
#retain_no 0.0
#update 0.0
#other 0.17857142857142858

#python3 bioasq_evaluation.py \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_context_prompt1_exp2_bioasq.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_nocontext_prompt1_exp2_bioasq.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/chatgpt_0shot_nocontext_prompt1_exp2_bioasq_comp.jsonl
#


#python3 bioasq_evaluation.py \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/gpt_0shot_biqasq_context.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/gpt_0shot_biqasq_nocontext.jsonl \
#/Users/wenbingbing/PycharmProjects/qasper/results/bioasq/gpt_0shot_biqasq_nocontext_comp.jsonl