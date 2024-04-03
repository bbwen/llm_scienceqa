python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \


python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \

python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random

python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double



python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \


python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --no_context \

python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 8 \
    --random

python run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 8 \
    --random \
    --double




python3 run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --openai_engine "gpt-3.5-turbo-0613" \
    --eval_batch_size 8 \


python3 run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --openai_engine "gpt-3.5-turbo-0613" \
    --eval_batch_size 8 \
    --no_context \

python3 run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results/squad  \
    --openai_engine "gpt-3.5-turbo-0613" \
    --eval_batch_size 8 \
    --random

python3 run_eval.py \
    --dataset squad2 \
    --dataset_path "" \
    --prompt_key sota \
    --save_dir  ./results/squad \
    --openai_engine "gpt-3.5-turbo-0613" \
    --eval_batch_size 8 \
    --random \
    --double



# python run_eval.py \
#     --save_dir  /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/squad2/open/llama/no_context \
#     --model_name_or_path /home/notebook/data/group/Ziwei/LLM/LLaMA_v2/Llama-2-13b-chat-hf/ \
#     --tokenizer_name_or_path /home/notebook/data/group/Ziwei/LLM/LLaMA_v2/Llama-2-13b-chat-hf/ \
#     --use_chat_format \
#     --eval_batch_size 32 \
    # --no_context


# python run_eval.py \
#     --save_dir  /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/squad2/open/vicuna/nshot \
#     --model_name_or_path  /home/notebook/data/group/Ziwei/LLM/Vicuna/vicuna-13b-v1.5 \
#     --tokenizer_name_or_path /home/notebook/data/group/Ziwei/LLM/Vicuna/vicuna-13b-v1.5 \
#     --use_chat_format \
#     --eval_batch_size 16 \
#     --n_shot 3  # need to decrease batchsize to 16 for n_shot
#     # --no_context



#python run_eval.py \
#    --save_dir  /home/notebook/data/personal/S9053169/llm_scienceqa/qasper/results/squad2/open/flan \
#    --model_name_or_path  /home/notebook/data/personal/S9053169/llm_scienceqa/flan-t5-xxl \
#    --tokenizer_name_or_path /home/notebook/data/personal/S9053169/llm_scienceqa/flan-t5-xxl \
#    --use_chat_format \
#    --eval_batch_size 32 \
#    --flan_model
#    # --no_context