import os

from file_utils import load_jsonl
import tiktoken

INSTRUCTION_STR = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_INSTRUCTION_STR = "Give simple short one phrase answers for the question"


from transformers import LlamaTokenizer, T5Tokenizer

BASE_FOLDER = os.getenv('DBQA')
DATA_FOLDER = f"{BASE_FOLDER}/data"
RETRIEVER_FOLDER = f"{BASE_FOLDER}/retriever_results"
READER_FOLDER = f"{BASE_FOLDER}/reader_results"

complete_model_names = {
   "llama2_7b": "huggingface/meta-llama/Llama-2-7b-hf",
    "llama2_70b": "huggingface/meta-llama/Llama-2-70b-hf",
    "flanT5": "huggingface/google/flan-t5-xxl",
    "flanUl2": "huggingface/google/flan-ul2",
    "gpt-3.5": "gpt-3.5-turbo-0125"
}

dataset_map = {
    "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
    "nq": "nq-dev-kilt.jsonl",
    "bioasq": "bioasq.jsonl",
}

tokenizer_path_map = {
    "llama2_7b": "/data/datasets/models/meta-ai/llama2/weights/",
    "llama2_70b": "/data/datasets/models/meta-ai/llama2/weights/",
    "flanT5": "google/flan-ul2",
    "flanUl2": "google/flan-ul2"
}

tokenizer_map = {
    "llama2_7b": LlamaTokenizer,
    "llama2_70b": LlamaTokenizer,
    "flanT5": T5Tokenizer,
    "flanUl2": T5Tokenizer
}


def get_tokenizer(model_name):
    if model_name in tokenizer_map and model_name in tokenizer_path_map:
        return tokenizer_map[model_name].from_pretrained(tokenizer_path_map[model_name])
    else:
        raise Exception(f"{model_name} is not supported. Add the corresponding tokenizer and tokenizer_path in utils")

def truncate_prompt(prompt, tokenizer, instruction_str_tokens, total_tokens, max_new_tokens):
    format_tokens =  tokenizer("\nContext: \nQuestion: \nAnswer:")
    question_tokens = tokenizer(prompt["question"])["input_ids"]
    remaining_length = total_tokens-len(format_tokens)-len(instruction_str_tokens)-len(question_tokens)-max_new_tokens-10 #additional buffer of 5
    context_tokens_before_truncation = tokenizer(prompt["context"], add_special_tokens=False)["input_ids"]
    context_tokens_after_truncation = tokenizer(prompt["context"], max_length=remaining_length, truncation=True, add_special_tokens=False)["input_ids"]
    context_str_after_truncation = tokenizer.decode(context_tokens_after_truncation)
    context_length_change_info = {
        "original_context_str_length": len(prompt["context"]),
        "context_str_length_after_truncation": len(context_str_after_truncation),
        "original_context_token_length": len(context_tokens_before_truncation),
        "context_token_length_after_truncation": len(context_tokens_after_truncation)
    }
    modified_prompt = create_prompt(question=prompt["question"], context=context_str_after_truncation)
    return modified_prompt, context_length_change_info

def post_process_answers(answers):
    modified_answers = []
    for x in answers:
        if not x:
            modified_answers.append("")
        else:
            modified_answers.append(x.strip().split("\n")[0])
    return modified_answers

def create_prompt(question, context):
    if context:
        return f"{INSTRUCTION_STR}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_INSTRUCTION_STR}\nQuestion: {question}\nAnswer: ".strip()

def num_gpt_tokens_per_message(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_gpt_tokens_per_message(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_gpt_tokens_per_message(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    # if (self_enclosed):
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def num_gpt_tokens_per_content(content, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # num_tokens = 0
    encoded_content = encoding.encode(content)
    num_tokens = len(encoded_content)
    return num_tokens, encoded_content


def merge_retriever_data_and_eval_results(retriever_data_path, retriever_eval_data_path):
    if os.path.exists(retriever_data_path) and os.path.exists(retriever_eval_data_path):
        retriever_data = load_jsonl(retriever_data_path)
        retriever_eval_data = load_jsonl(retriever_eval_data_path)
        for retriever_info, eval_info in zip(retriever_data, retriever_eval_data):
            for r,e in zip(retriever_info["output"][0]["provenance"], eval_info.get("doc-level results", eval_info.get("passage-level results"))):
                r["page_par_id_match"] = e.get("pm_sec_id_match", e.get("wiki_par_id_match", e.get("page_par_id_match")))
        return retriever_data
    elif os.path.exists(retriever_data_path):
        return load_jsonl(retriever_data_path)
    else:
        raise Exception(f"{retriever_data_path} does not exist!")
