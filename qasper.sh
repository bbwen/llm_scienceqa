#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --flan_model
#
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --no_context \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --random \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#    --flan_model
#
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl  \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --flan_model
#
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --no_context \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --random \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#    --flan_model




#

python run_eval.py \
    --dataset qasper \
    --dataset_path qasper-test-v0.3.json \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --use_vllm


#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random

python run_eval.py \
    --dataset qasper \
    --dataset_path qasper-test-v0.3.json \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --use_vllm \
    --random \
    --double


#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random \
#    --double


python run_eval.py \
    --dataset qasper \
    --dataset_path qasper-test-v0.3.json \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 1 \
    --use_vllm \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random

python run_eval.py \
    --dataset qasper \
    --dataset_path qasper-test-v0.3.json \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 1 \
    --use_vllm \
    --random \
    --double


#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5  \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_path qasper-test-v0.3.json \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --use_vllm \
#    --random \
#    --double