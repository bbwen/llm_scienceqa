export CUDA_VISIBLE_DEVICES=1



#flan
python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 8 \
    --flan_model \


python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double \
    --flan_model \



python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \
    --flan_model \

python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path google/flan-t5-xl \
    --tokenizer_name_or_path google/flan-t5-xl  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --flan_model \





#llama
python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \


python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \

python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random

python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double


#vicuna
python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \


python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \

python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --random

python run_eval.py \
    --dataset squad2 \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double




#python3 run_eval.py \
#    --dataset squad2 \
#    --dataset_path "" \
#    --prompt_key sota \
#    --save_dir  ./results/squad \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#
#
#python3 run_eval.py \
#    --dataset squad2 \
#    --dataset_path "" \
#    --prompt_key sota \
#    --save_dir  ./results/squad \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --no_context \
#
#python3 run_eval.py \
#    --dataset squad2 \
#    --dataset_path "" \
#    --prompt_key sota \
#    --save_dir  ./results/squad  \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random
#
#python3 run_eval.py \
#    --dataset squad2 \
#    --dataset_path "" \
#    --prompt_key sota \
#    --save_dir  ./results/squad \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random \
#    --double


