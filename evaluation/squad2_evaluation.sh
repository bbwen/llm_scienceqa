 echo "no context sota llama"
 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 squad2_evaluator_gpt_extractive_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 squad2_evaluator_gpt_extractive_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \


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
 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl
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
python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

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
python3 squad2_evaluator_gpt_extractive_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##



echo "random context sota gpt"

python3 squad2_evaluator_gpt_extractive_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl
##




echo "double random context sota llama"
 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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
 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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

 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl





echo "double random context sota gpt"
 python3 squad2_evaluator_gpt_extractive_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl

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












####noisy context
#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_Llama-2-13b-chat-hf_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

#hashas 0.6254208754208754
#hasno 0.010101010101010102
#nohas 0.23737373737373738
#nono 0.1271043771043771
#{
#  "Answer F1": 0.5541363837656927,
#  "Answer F1 by type": {
#    "noANS": 0.584873949579832,
#    "hasANS": 0.5232951499386904
#  },
#  "Missing predictions": 10685,
#  "Answerability": 0.36447811447811446,
#  "Noans": {
#    "noANS": 0.584873949579832,
#    "hasANS": 0.1433389544688027
#  },
#  "Answer F1 after": 0.3102149602785132,
#  "Answer F1 by type after": {
#    "noANS": 0.23865546218487396,
#    "hasANS": 0.3820158057518943
#  },
#  "Answerability after": 0.13720538720538722,
#  "Noans after": {
#    "noANS": 0.23865546218487396,
#    "hasANS": 0.03541315345699832
#  }
#}



#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

#hashas 0.484006734006734
#hasno 0.08838383838383838
#nohas 0.10353535353535354
#nono 0.32407407407407407
#{
#  "Answer F1": 0.6095535700544197,
#  "Answer F1 by type": {
#    "noANS": 0.6470588235294118,
#    "hasANS": 0.5719218233130712
#  },
#  "Missing predictions": 10685,
#  "Answerability": 0.4276094276094276,
#  "Noans": {
#    "noANS": 0.6470588235294118,
#    "hasANS": 0.20741989881956155
#  },
#  "Answer F1 after": 0.5752614007546226,
#  "Answer F1 by type after": {
#    "noANS": 0.5798319327731093,
#    "hasANS": 0.5706754537883505
#  },
#  "Answerability after": 0.41245791245791247,
#  "Noans after": {
#    "noANS": 0.5798319327731093,
#    "hasANS": 0.24451939291736932
#  }
#}


#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl

#hashas 0.6750841750841751
#hasno 0.03956228956228956
#nohas 0.04208754208754209
#nono 0.24326599326599327
#{
#  "Answer F1": 0.6016366852458388,
#  "Answer F1 by type": {
#    "noANS": 0.5327731092436975,
#    "hasANS": 0.6707325161417491
#  },
#  "Missing predictions": 10685,
#  "Answerability": 0.28535353535353536,
#  "Noans": {
#    "noANS": 0.5327731092436975,
#    "hasANS": 0.03709949409780776
#  },
#  "Answer F1 after": 0.5808932519683454,
#  "Answer F1 by type after": {
#    "noANS": 0.5109243697478991,
#    "hasANS": 0.6510981169281524
#  },
#  "Answerability after": 0.2828282828282828,
#  "Noans after": {
#    "noANS": 0.5109243697478991,
#    "hasANS": 0.05396290050590219
#  }
#}



###no context sota

python3 squad2_evaluator_gpt_extractive_comp.py \
--predictions  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl  \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/squad/squad2_gpt-3.5-turbo-0613_0_sota_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl





















#
#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_context_prompt1_exp2_squad2.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_nocontext_prompt1_exp2_squad2.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_nocontext_prompt1_exp2_squad2_comp.jsonl

#hashas 0.6470588235294118
#hasno 0.029411764705882353
#nohas 0.025210084033613446
#nono 0.29831932773109243
#{
#  "Answer F1": 0.6454208997461592,
#  "Answer F1 by type": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.7446233120973604
#  },
#  "Missing predictions": 11635,
#  "Answerability": 0.3235294117647059,
#  "Noans": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.10084033613445378
#  },
#  "Answer F1 after": 0.6144265586898986,
#  "Answer F1 by type after": {
#    "noANS": 0.5210084033613446,
#    "hasANS": 0.7078447140184527
#  },
#  "Answerability after": 0.3277310924369748,
#  "Noans after": {
#    "noANS": 0.5210084033613446,
#    "hasANS": 0.13445378151260504
#  }
#}




#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_context_prompt1_exp2_squad2.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_doublecontext_prompt1_exp2_squad2.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_doublecontext_prompt1_exp2_squad2_comp.jsonl

#{
#  "Answer F1": 0.6454208997461592,
#  "Answer F1 by type": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.7446233120973604
#  },
#  "Missing predictions": 11635,
#  "Answerability": 0.3235294117647059,
#  "Noans": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.10084033613445378
#  },
#  "Answer F1 after": 0.6114288139498224,
#  "Answer F1 by type after": {
#    "noANS": 0.48739495798319327,
#    "hasANS": 0.7354626699164514
#  },
#  "Answerability after": 0.3025210084033613,
#  "Noans after": {
#    "noANS": 0.48739495798319327,
#    "hasANS": 0.11764705882352941
#  }
#}


#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/results/squad2/1_squad2_chatgpt_context_zero_shot_ran101.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/squad2/1_squad2_chatgpt_nocontext_zero_shot_new_ran101.jsonl  \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/results/squad2/1_squad2_chatgpt_nocontext_zero_shot_new_ran101_comp.jsonl
#{
#  "Answer F1": 0.45135591590274704,
#  "Answer F1 by type": {
#    "noANS": 0.42016806722689076,
#    "hasANS": 0.482543764578603
#  },
#  "Missing predictions": 11635,
#  "Answerability": 0.226890756302521,
#  "Noans": {
#    "noANS": 0.42016806722689076,
#    "hasANS": 0.03361344537815126
#  },
#  "Answer F1 after": 0.49603847136713325,
#  "Answer F1 by type after": {
#    "noANS": 0.46218487394957986,
#    "hasANS": 0.529892068784686
#  },
#  "Answerability after": 0.24369747899159663,
#  "Noans after": {
#    "noANS": 0.46218487394957986,
#    "hasANS": 0.025210084033613446
#  }
#}





#python3 squad2_evaluator_gpt_extractive_comp.py \
#--predictions  /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_context_prompt1_exp2_squad2.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_nocontext_prompt1_exp2_squad2.jsonl \
#--comp_file /Users/wenbingbing/PycharmProjects/qasper/results/squad2/chatgpt_0shot_nocontext_prompt1_exp2_squad2_comp.jsonl
#exp2  F1 0.645
#{
#  "Answer F1": 0.6454208997461592,
#  "Answer F1 by type": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.7446233120973604
#  },
#  "Missing predictions": 11635,
#  "Answerability": 0.3235294117647059,
#  "Noans": {
#    "noANS": 0.5462184873949579,
#    "hasANS": 0.10084033613445378
#  },
#  "Answer F1 after": 0.6144265586898986,
#  "Answer F1 by type after": {
#    "noANS": 0.5210084033613446,
#    "hasANS": 0.7078447140184527
#  },
#  "Answerability after": 0.3277310924369748,
#  "Noans after": {
#    "noANS": 0.5210084033613446,
#    "hasANS": 0.13445378151260504
#  }
#}


