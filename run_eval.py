import argparse
import os
from os import path, mkdir, getenv
import json
import random
import torch
#import vllm
from statistics import mean
from huggingface_hub import login as hf_login
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from templates import create_prompt_with_llama2_chat_format
from dataset_loader import load_data
from utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
)


encoding_templates_with_context = {
    "bioasq":{
    "sota": ("Create an \"Answer\" to the \"Question\" using following documents. Pay attention to answer only \"yes\" and \"no\"\n\n", "Documents:", "Question:", "Answer:"),
    "freeform": ("Create an \"Answer\" to the \"Question\" using following documents.\n\n", "Documents:", "Question:", "Answer:")},
    "pubmedqa":{
    "sota": ("Create an \"Answer\" to the \"Question\" using following documents. Pay attention to answer only \"yes\"，\"no\" and \"Unanswerable\". Answer \"Unanswerable\" when you are not sure about the answer. \n\n", "Documents:", "Question:", "Answer:"),
    "freeform": ( "Create an \"Answer\" to the \"Question\" use following documents. Answer \"Unanswerable\" when you are not sure about the answer. \n\n","Documents:", "Question:", "Answer:")},
    "qasper": {
        "sota": ("Create an \"Answer\" to the \"Question\" using following documents. Pay attention to answer only \"yes\" or \"no\" for boolean questions. Answer \"Unanswerable\" when you are not sure about the answer.Please only output the exact answer and keep the answer concise\n\n", "Documents:", "Question:", "Answer:"),
        "freeform": ("Create an \"Answer\" to the \"Question\" using the following documents. Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n", "Documents:", "Question:", "Answer:")},
    "squad2": {
        "sota": ("Create a shortest \"Answer\" to the \"Question\" using the following documents. Answer \"Unanswerable\" when you are not sure about the answer.\n\n", "Documents:", "Question:", "Answer:"),
        "freeform": ( "Create a shortest \"Answer\" to the \"Question\" using the following documents. Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n",
        "Documents:", "Question:", "Answer:")},
}

encoding_templates_without_context = {
    "bioasq": {
    "sota": ("create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\" and \"no\". Answer \"Unanswerable\" when you are not sure about the answer.\n\n", "Question:", "Answer:"),
    "freeform": ("create an \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer.\n\n",
    "Question:", "Answer:")},
    "pubmedqa":{
    "sota": ("Create an \"Answer\" to the \"Question\". Pay attention to answer only \"yes\"，\"no\" and \"Unanswerable\". Answer \"Unanswerable\" when you are not sure about the answer.\n\n", "Question:", "Answer:"),
    "freeform": ( "Create an \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. \n\n", "Question:", "Answer:")
    },
    "qasper": {
        "sota": (
        "Create an \"Answer\" to the \"Question\". Answer \"yes\" or \"no\" for boolean questions. Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n", "Question:", "Answer:"),
        "freeform": (
        "Create an \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n", "Question:", "Answer:")},

    "squad2": {
        "sota": ("Create a shortest \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n", "Question:", "Answer:"),
        "freeform": ("Create a shortest \"Answer\" to the \"Question\". Answer \"Unanswerable\" when you are not sure about the answer. Please only output the exact answer and keep the answer concise\n\n", "Question:", "Answer:")},
}

def main(args):
    random.seed(101)

    # HF Login


    print("Loading data...")
    data = load_data(args.dataset, args.dataset_dir)
    print(len(data))
    test_data = []
    context_length_list = []
    truncated_list = []
    for item in list(data.keys()):
        question = data[item]['question']
        context = data[item]['context']
        gold_answer = data[item]['answer']
        random_context = data[item]['random_context']
        if args.double and args.random:
            context = context + random_context
        else:
            if args.double:
                if len(context) > 1:
                    random_sample = random.sample(context, k=1)
                    context = context + random_sample
                else:
                    context = context + context
            if args.random:
                context = random_context

        example = {
            "id": item,
            "context": "\n".join(context),
            "question": question,
            "answers": gold_answer
        }
        test_data.append(example)


    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            tokenizer = model.llm_engine.tokenizer
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                flan_model= args.flan_model,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
    else:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # reduce context length to max_context_length
    if args.max_context_length:
        for example in test_data:
            tokenized_context = tokenizer.encode(example["context"])
            if len(tokenized_context) >0:
                context_length_list.append(len(tokenized_context))
                if len(tokenized_context) >args.max_context_length:
                    truncated_list.append(len(tokenized_context))
            if len(tokenized_context) > args.max_context_length:
                example["context"] = tokenizer.decode(tokenized_context[:args.max_context_length])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prompts = []
    for example in test_data:
        if args.no_context:
            prompt, q_template, a_template = encoding_templates_without_context[args.dataset][args.prompt_key]
            p_template = ""
        else:
            prompt, p_template, q_template, a_template = encoding_templates_with_context[args.dataset][args.prompt_key]

        prompt += "\n\n"

        if args.no_context:
            prompt += q_template + " " + format(example["question"]) + "\n"
        else:
            prompt += p_template + " " + format(example["context"]) + "\n" + q_template + " " + format(example["question"]) + "\n"

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = create_prompt_with_llama2_chat_format(messages, add_bos=False)
            prompt += a_template if prompt[-1] in ["\n", " "] else " " + a_template
        else:
            prompt += a_template
        prompts.append(prompt)

    if args.model_name_or_path:
        if args.use_vllm:
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=256,
                stop=["\n"] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
            )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt].strip() if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=256,
                batch_size=args.eval_batch_size,
                flan_model = args.flan_model,
                stop_id_sequences=[[new_line_token]] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
            )
            # remove unnecessary space
            outputs = [output.strip() for output in outputs]
    else:
        instances = [{"id": example["id"], "prompt": prompt} for example, prompt in zip(test_data, prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=os.path.join(args.save_dir, f"{args.dataset}_{args.openai_engine}_{args.n_shot}_{args.prompt_key}_results_nocontext{args.no_context}_random{args.random}_double{args.double}.jsonl"),
            batch_size=args.eval_batch_size,
        )
        outputs = [result["output"].strip().split("\n")[0].strip() for result in results]
    if args.model_name_or_path:
        model_name= args.model_name_or_path.split("/")[-1]
    else:
        model_name = args.openai_engine
    print(mean(context_length_list))
    print(len(truncated_list)/len(context_length_list))
    with open(os.path.join(args.save_dir, f"{args.dataset}_{model_name}_{args.n_shot}_{args.prompt_key}_results_nocontext{args.no_context}_random{args.random}_double{args.double}.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            each = {"question_id": example["id"], "question": example["question"], "predicted_answer": output, "gold_answer": example["answers"], "predicted_evidence":example["context"]}
            fout.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="test.json"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="bioasq"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
        help="number of examples to use for few-shot evaluation."
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="sota",
        help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=2048,
        help="maximum number of tokens in the context passage."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--flan_model",
        action="store_true",
        help="If given, we're evaluating a flan model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_llama2_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')

    args = parser.parse_args()

    if args.hf_token_var and (args.openai_engine is None):
        hf_login(token=getenv(args.hf_token_var))
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
