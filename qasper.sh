export CUDA_VISIBLE_DEVICES=0

#flan


#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 16 \
#    --random \
#    --double \
#    --flan_model
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 16 \
#    --no_context \
#    --flan_model
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 16 \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 16 \
#    --random \
#    --flan_model
#





#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl  \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --flan_model
#
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --no_context \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --random \
#    --flan_model
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#    --flan_model




#llama
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 8 \
#    --random \
#    --double
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 8 \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --use_chat_format \
#    --eval_batch_size 8 \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key sota \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 8 \
#    --random




#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 2 \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --random
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --random \
#    --double


#vicuan

python run_eval.py \
    --dataset qasper \
    --dataset_dir ./data/qasper/ \
    --prompt_key sota \
    --save_dir  ./results/qasper \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double

python run_eval.py \
    --dataset qasper \
    --dataset_dir ./data/qasper/ \
    --prompt_key sota \
    --save_dir  ./results/qasper \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \
#

python run_eval.py \
    --dataset qasper \
    --dataset_dir ./data/qasper/ \
    --prompt_key sota \
    --save_dir  ./results/qasper \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \

python run_eval.py \
    --dataset qasper \
    --dataset_dir ./data/qasper/ \
    --prompt_key sota \
    --save_dir  ./results/qasper \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --random




#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path lmsys/vicuna-13b-v1.5  \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 2 \
#
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --no_context \
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --random
#
#python run_eval.py \
#    --dataset qasper \
#    --dataset_dir ./data/qasper/ \
#    --prompt_key freeform \
#    --save_dir  ./results/qasper \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 2 \
#    --random \
#    --double