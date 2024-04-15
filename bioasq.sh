export CUDA_VISIBLE_DEVICES=0


####flan

#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#    --flan_model \
#    --max_context_length 2048 \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --flan_model \
#    --max_context_length 2048 \
#
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --use_chat_format \
#    --eval_batch_size 8 \
#    --no_context \
#    --flan_model \
#    --max_context_length 2048 \
#
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key sota \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --random \
#    --flan_model \
#    --max_context_length 2048 \





#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl  \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --flan_model \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --no_context \
#    --flan_model \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl  \
#    --eval_batch_size 8 \
#    --random \
#    --flan_model \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path google/flan-t5-xl \
#    --tokenizer_name_or_path google/flan-t5-xl \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#    --flan_model \

#llama

python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 4 \
    --random \
    --double \
    --max_context_length 2048 \

python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 4 \
    --max_context_length 2048 \



python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --use_chat_format \
    --eval_batch_size 4 \
    --no_context \
    --max_context_length 2048 \


python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
    --use_chat_format \
    --eval_batch_size 4 \
    --random \
    --max_context_length 2048 \





#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --no_context \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --random
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#    --tokenizer_name_or_path meta-llama/Llama-2-13b-chat-hf  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --random \
#    --double


#vicuna
python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 4 \
    --max_context_length 2048 \



python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 4 \
    --no_context \
    --max_context_length 2048 \


python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
    --use_chat_format \
    --eval_batch_size 4 \
    --random \
    --max_context_length 2048 \


python run_eval.py \
    --dataset bioasq \
    --dataset_dir  ./data/bioasq/ \
    --prompt_key sota \
    --save_dir  ./results \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
    --use_chat_format \
    --eval_batch_size 4 \
    --random \
    --double \
    --max_context_length 2048 \


#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5  \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --no_context \
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5  \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --random
#
#python run_eval.py \
#    --dataset bioasq \
#    --dataset_dir  ./data/bioasq/ \
#    --prompt_key freeform \
#    --save_dir  ./results \
#    --model_name_or_path lmsys/vicuna-13b-v1.5 \
#    --tokenizer_name_or_path lmsys/vicuna-13b-v1.5 \
#    --use_chat_format \
#    --eval_batch_size 1 \
#    --random \
#    --double




###chatgpt

#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json  \
#    --prompt_key sota \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key sota \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --no_context \
#
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key sota \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random \
#
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key sota \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random \
#    --double \
#
#
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key freeform \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key freeform \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --no_context \
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key freeform \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random \
#
#python3 run_eval.py \
#    --dataset bioasq \
#    --dataset_path /Users/wenbingbing/PycharmProjects/qasper/data/bioasq/test.json \
#    --prompt_key freeform \
#    --save_dir  ./results215 \
#    --openai_engine "gpt-3.5-turbo-0613" \
#    --eval_batch_size 8 \
#    --random \
#    --double \

