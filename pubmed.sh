export CUDA_VISIBLE_DEVICES=1

#llama
python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/  \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


#vicuna
python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5  \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


#flanxl
python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key sota \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl  \
    --tokenizer_name_or_path google/flan-t5-xl \
    --use_chat_format \
    --eval_batch_size 1 \


python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl \
    --use_chat_format \
    --eval_batch_size 1 \
    --no_context \

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 1 \
    --random

python run_eval.py \
    --dataset pubmedqa \
    --dataset_dir ./data/pubmedqa/ \
    --prompt_key freeform \
    --save_dir  ./results/pubmedqa \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl \
    --use_chat_format \
    --eval_batch_size 1 \
    --random \
    --double


#python3 run_eval.py \
#    --dataset pubmedqa \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/pubmedqa/pub_test.json \
#    --prompt_key sota \
#    --save_dir  ./results/pubmedqa \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#
#python3 run_eval.py \
#    --dataset pubmedqa \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/pubmedqa/pub_test.json \
#    --prompt_key freeform \
#    --save_dir  ./results/pubmedqa \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \