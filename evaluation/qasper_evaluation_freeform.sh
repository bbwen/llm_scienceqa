 echo "no context sota llama freeform"
python3 qasper_evaluator_flan_comp.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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
python3 qasper_evaluator_flan_comp.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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
python3 qasper_evaluator_flan_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only

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
 python3 qasper_evaluator_flan_comp.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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
python3 qasper_evaluator_flan_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only

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
python3 qasper_evaluator_flan_comp.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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
python3 qasper_evaluator_flan_comp.py  \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
##



echo "random context sota gpt"

python3 qasper_evaluator_flan_comp.py \
--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleFalse.jsonl \
--comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleFalse_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
##




echo "double random context sota llama"
 python3 qasper_evaluator_flan_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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
 python3 qasper_evaluator_flan_comp.py  \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only
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

 python3 qasper_evaluator_flan_comp.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only




echo "double random context sota gpt"
 python3 qasper_evaluator_flan_comp.py \
 --predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
 --predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
 --comp_file /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_gpt-3.5-turbo-0613_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
--text_evidence_only





####no context sota
#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_sota_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--text_evidence_only


#{'Answer F1': 0.4663823540505264, 'Answer F1 by type': {'extractive': 0.40509203486067474, 'abstractive': 0.15439078021919075, 'boolean': 0.7635467980295566, 'none': 0.740506329113924}, 'Answerability_all': 0.16083916083916083, 'Answerability_has': 0.07971656333038087, 'noans F1 by type': {'extractive': 0.07983761840324763, 'abstractive': 0.1657754010695187, 'boolean': 0.0, 'none': 0.740506329113924}, 'Answer F1 after': 0.22038839881977138, 'Answer F1 by type after': {'extractive': 0.0, 'abstractive': 0.0030469965764083414, 'boolean': 0.6039603960396039, 'none': 0.8429319371727748}, 'Answerability_all_after': 0.7894327894327894, 'Answerability_has_after': 0.7801094890510949, 'Noans after': {'extractive': 0.9736842105263158, 'abstractive': 0.9, 'boolean': 0.0, 'none': 0.8429319371727748}}



###no context freeform

#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_freeform_results_nocontextTrue_randomFalse_doubleFalse.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
#--text_evidence_only

#{'Answer F1': 0.6064549200439114, 'Answer F1 by type': {'extractive': 0.601724715905484, 'abstractive': 0.2768152351831215, 'boolean': 0.5683060109289617, 'none': 0.9779005524861878}, 'Answerability_all': 0.2680652680652681, 'Answerability_has': 0.1518987341772152, 'noans F1 by type': {'extractive': 0.10785619174434088, 'abstractive': 0.2558139534883721, 'boolean': 0.23497267759562843, 'none': 0.9779005524861878}, 'Answer F1 after': 0.18026418026418026, 'Answer F1 by type after': {'extractive': 0.0, 'abstractive': 0.0, 'boolean': 0.04819277108433735, 'none': 0.986784140969163}, 'Answerability_all_after': 0.9875679875679876, 'Answerability_has_after': 0.9877358490566037, 'Noans after': {'extractive': 0.9985380116959064, 'abstractive': 1.0, 'boolean': 0.927710843373494, 'none': 0.986784140969163}}





## distract context free-form
#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
#--text_evidence_only

#hashas 0.7249417249417249
#hasno 0.07925407925407925
#nohas 0.08080808080808081
#nono 0.11499611499611499
#{'Answer F1': 0.4566192445906289, 'Answer F1 by type': {'extractive': 0.4003043341683691, 'abstractive': 0.2750613808161976, 'boolean': 0.6080121770394573, 'none': 0.7919463087248322}, 'Answerability_all': 0.1958041958041958, 'Answerability_has': 0.11775043936731107, 'noans F1 by type': {'extractive': 0.07940780619111709, 'abstractive': 0.15196078431372548, 'boolean': 0.23036649214659685, 'none': 0.7919463087248322}, 'Answer F1 after': 0.4227092105789762, 'Answer F1 by type after': {'extractive': 0.3485391136219781, 'abstractive': 0.2518297943256608, 'boolean': 0.579538554948391, 'none': 0.81875}, 'Answerability_all_after': 0.19425019425019424, 'Answerability_has_after': 0.10559006211180125, 'Noans after': {'extractive': 0.0824022346368715, 'abstractive': 0.10964912280701754, 'boolean': 0.1912568306010929, 'none': 0.81875}}

#
#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_sota_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
#--text_evidence_only
#hashas 0.5252525252525253
#hasno 0.14296814296814297
#nohas 0.10178710178710179
#nono 0.22999222999222999
#{'Answer F1': 0.30489845618189, 'Answer F1 by type': {'extractive': 0.20869776880937865, 'abstractive': 0.0784813517347127, 'boolean': 0.654639175257732, 'none': 0.5698324022346368}, 'Answerability_all': 0.3317793317793318, 'Answerability_has': 0.2933212996389892, 'noans F1 by type': {'extractive': 0.3053977272727273, 'abstractive': 0.3761904761904762, 'boolean': 0.15979381443298968, 'none': 0.5698324022346368}, 'Answer F1 after': 0.46187678053381864, 'Answer F1 by type after': {'extractive': 0.3597130021504212, 'abstractive': 0.14910061399345037, 'boolean': 0.708994708994709, 'none': 0.9312169312169312}, 'Answerability_all_after': 0.372960372960373, 'Answerability_has_after': 0.2768670309653916, 'Noans after': {'extractive': 0.2828854314002829, 'abstractive': 0.3811881188118812, 'boolean': 0.14285714285714285, 'none': 0.9312169312169312}}


#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_vicuna-13b-v1.5_0_freeform_results_nocontextFalse_randomTrue_doubleTrue_comp.jsonl \
#--text_evidence_only

#hashas 0.7404817404817405
#hasno 0.11033411033411034
#nohas 0.03418803418803419
#nono 0.11499611499611499
#{'Answer F1': 0.5407309919619205, 'Answer F1 by type': {'extractive': 0.5731988489195523, 'abstractive': 0.31450018076181296, 'boolean': 0.6565656565656566, 'none': 0.5298013245033113}, 'Answerability_all': 0.14918414918414918, 'Answerability_has': 0.09859154929577464, 'noans F1 by type': {'extractive': 0.08130081300813008, 'abstractive': 0.175, 'boolean': 0.08585858585858586, 'none': 0.5298013245033113}, 'Answer F1 after': 0.5513780633118276, 'Answer F1 by type after': {'extractive': 0.531558416513403, 'abstractive': 0.29034178346814477, 'boolean': 0.6752577319587629, 'none': 0.8159509202453987}, 'Answerability_all_after': 0.22533022533022534, 'Answerability_has_after': 0.1396797153024911, 'Noans after': {'extractive': 0.12362637362637363, 'abstractive': 0.23267326732673269, 'boolean': 0.10309278350515463, 'none': 0.8159509202453987}}
#


#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_sota_results_nocontextFalse_randomFalse_doubleFalse.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_flan-t5-xl_0_sota_results_nocontextFalse_randomTrue_doubleTrue.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/science_qa_new/results/qasper/qasper_Llama-2-13b-chat-hf_0_freeform_results_nocontextTrue_randomFalse_doubleFalse_comp.jsonl \
#--text_evidence_only

#hashas 0.8228438228438228
#hasno 0.016317016317016316
#nohas 0.03341103341103341
#nono 0.12742812742812742
#
#{'Answer F1': 0.4663823540505264, 'Answer F1 by type': {'extractive': 0.40509203486067474, 'abstractive': 0.15439078021919075, 'boolean': 0.7635467980295566, 'none': 0.740506329113924}, 'Answerability_all': 0.16083916083916083, 'Answerability_has': 0.07971656333038087, 'noans F1 by type': {'extractive': 0.07983761840324763, 'abstractive': 0.1657754010695187, 'boolean': 0.0, 'none': 0.740506329113924}, 'Answer F1 after': 0.46958619626208875, 'Answer F1 by type after': {'extractive': 0.4132154821948446, 'abstractive': 0.13131018101432718, 'boolean': 0.8106796116504854, 'none': 0.6967741935483871}, 'Answerability_all_after': 0.14374514374514374, 'Answerability_has_after': 0.06802120141342756, 'Noans after': {'extractive': 0.06648575305291723, 'abstractive': 0.14814814814814814, 'boolean': 0.0, 'none': 0.6967741935483871}}
#


















#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/results/qasper/qasper_chatgpt_top_p1_0shot_context_random101.jsonl \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/qasper/qasper_chatgpt_top_p1_0shot_nocontext_new_100.jsonl  \
#--gold /Users/wenbingbing/PycharmProjects/qasper/data/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/results/qasper/qasper_chatgpt_top_p1_0shot_nocontext_new_100_comp.jsonl \
#--text_evidence_only
#hashas 0.045871559633027525
#hasno 0.6467889908256881
#nohas 0.013761467889908258
#nono 0.29357798165137616
#{'Answer F1': 0.5089590880050379, 'Answer F1 by type': {'extractive': 0.48186503498460415, 'abstractive': 0.16386769630813242, 'boolean': 0.6428571428571429, 'none': 1.0}, 'Evidence F1': 0.28440366972477066, 'Answerability': 0.3073394495412844, 'Missing predictions': 1233, 'noans F1 by type': {'extractive': 0.10476190476190476, 'abstractive': 0.3137254901960784, 'boolean': 0.21428571428571427, 'none': 1.0}, 'Answer F1 after': 0.21910843195903457, 'Answer F1 by type after': {'extractive': 0.04958949096880131, 'abstractive': 0.06607965431494843, 'boolean': 0.25, 'none': 0.946725860155383}, 'Answerability after': 0.9403669724770642, 'Noans after': {'extractive': 0.9801980198019802, 'abstractive': 0.9375, 'boolean': 0.75, 'none': 0.975609756097561}}
#

#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/results/qasper/qasper_gpt_top_p1_0shot_context_random101.jsonl  \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/qasper/chatgpt_0shot_context__exp2_doc_qasper.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/data/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/results/qasper/chatgpt_0shot_context_exp2_doc_qasper_comp.jsonl  \
#--text_evidence_only
#{'Answer F1': 0.4980994398464396, 'Answer F1 by type': {'extractive': 0.45868061711650915, 'abstractive': 0.16813732555453417, 'boolean': 0.7193877551020408, 'none': 0.9666666666666667}, 'Evidence F1': 0.28440366972477066, 'Answerability': 0.30275229357798167, 'Missing predictions': 1233, 'noans F1 by type': {'extractive': 0.125, 'abstractive': 0.3541666666666667, 'boolean': 0.21428571428571427, 'none': 0.9666666666666667}, 'Answer F1 after': 0.4090653084106051, 'Answer F1 by type after': {'extractive': 0.4222802262331176, 'abstractive': 0.10165682337557337, 'boolean': 0.5, 'none': 0.7667108124458397}, 'Answerability after': 0.38073394495412843, 'Noans after': {'extractive': 0.17592592592592593, 'abstractive': 0.5, 'boolean': 0.41379310344827586, 'none': 0.8709677419354839}}

#python3 qasper_evaluator_flan_comp.py \
#--predictions /Users/wenbingbing/PycharmProjects/qasper/results/qasper/chatgpt_0shot_context__exp2_doc_qasper.jsonl  \
#--predictions_after /Users/wenbingbing/PycharmProjects/qasper/results/qasper/chatgpt_0shot_context_exp2_doc_boolfinal_qasper.jsonl \
#--gold /Users/wenbingbing/PycharmProjects/qasper/data/qasper/qasper-test-v0.3.json \
#--comp_file  /Users/wenbingbing/PycharmProjects/qasper/results/qasper/chatgpt_0shot_context_exp2_doc_boolfinal_qasper_comp.jsonl  \
#--text_evidence_only
#0,409 -> 0.454
#hashas 0.536697247706422
#hasno 0.08256880733944955
#nohas 0.09174311926605505
#nono 0.2889908256880734
#{'Answer F1': 0.4090653084106051, 'Answer F1 by type': {'extractive': 0.399988968245183, 'abstractive': 0.09954857326064222, 'boolean': 0.4827586206896552, 'none': 0.8709677419354839}, 'Evidence F1': 0.28440366972477066, 'Answerability': 0.38073394495412843, 'Missing predictions': 1233, 'noans F1 by type': {'extractive': 0.17592592592592593, 'abstractive': 0.5, 'boolean': 0.41379310344827586, 'none': 0.8709677419354839}, 'Answer F1 after': 0.4544503300728638, 'Answer F1 by type after': {'extractive': 0.49696598112142126, 'abstractive': 0.18277041195890775, 'boolean': 0.4827586206896552, 'none': 0.7180427547363031}, 'Answerability after': 0.37155963302752293, 'Noans after': {'extractive': 0.1651376146788991, 'abstractive': 0.52, 'boolean': 0.4482758620689655, 'none': 0.8}}
