 echo "no context sota llama"
 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file compare/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 evaluation/squad2_evaluation.py  \
--predictions results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file compare/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 evaluation/squad2_evaluation.py  \
--predictions results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after results/squad/squad2_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file compare/squad/squad2_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \


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
 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file compare/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file compare/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file compare/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 evaluation/squad2_evaluation.py  \
--predictions results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file compare/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##



echo "random context sota gpt"

python3 evaluation/squad2_evaluation.py  \
--predictions results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file compare/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##




echo "double random context sota llama"
 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file compare/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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
 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file compare/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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

 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file compare/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl





echo "double random context sota gpt"
 python3 evaluation/squad2_evaluation.py  \
 --predictions results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file compare/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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




























