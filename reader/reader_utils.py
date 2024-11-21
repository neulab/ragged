import os
import pdb
from file_utils import load_jsonl
import tiktoken
from litellm import token_counter

INSTRUCTION_STR = "Give simple short one phrase answers for the questions based on the context"
COT_INSTRUCTION_STR = "Answer the question based on the context passages provided. First, explain your reasoning step-by-step. Conclude with a simple, short answer in the format \"The answer is [ANSWER].\""
NO_CONTEXT_INSTRUCTION_STR = "Give simple short one phrase answers for the question"
NO_CONTEXT_COT_INSTRUCTION_STR = "First, explain your reasoning step-by-step. Conclude with a simple, short answer in the format \"The answer is [ANSWER].\""

context_instruction_dict = {
    "default": INSTRUCTION_STR,
    "cot": COT_INSTRUCTION_STR
}

no_context_instruction_dict = {
    "default": NO_CONTEXT_INSTRUCTION_STR,
    "cot": NO_CONTEXT_COT_INSTRUCTION_STR
}


from transformers import LlamaTokenizer, T5Tokenizer

# def get_tokenizer(model_name):
#     if model_name in tokenizer_map and model_name in tokenizer_path_map:
#         return tokenizer_map[model_name].from_pretrained(tokenizer_path_map[model_name])
#     else:
#         raise Exception(f"{model_name} is not supported. Add the corresponding tokenizer and tokenizer_path in utils")

# def get_token_len(model_name, text):
#     messages = [{"role": "user", "content": text}]
#     return token_counter(model=model_name, messages=messages)
    
# def truncate_prompt(prompt, tokenizer, total_tokens, max_new_tokens, prompt_mode):
#     format_tokens =  tokenizer("\nContext: \nQuestion: \nAnswer:")
#     question_tokens = tokenizer(prompt["question"])["input_ids"]
#     if prompt['context']:
#         instruction_str_tokens = tokenizer(context_instruction_dict[prompt_mode])["input_ids"]
#     else:
#         instruction_str_tokens = tokenizer(no_context_instruction_dict[prompt_mode])["input_ids"]
#     remaining_length = total_tokens-len(format_tokens)-len(instruction_str_tokens)-len(question_tokens)-max_new_tokens-15 #additional buffer
#     context_tokens_before_truncation = tokenizer(prompt["context"], add_special_tokens=False)["input_ids"]
#     context_tokens_after_truncation = tokenizer(prompt["context"], max_length=remaining_length, truncation=True, add_special_tokens=False)["input_ids"]
#     context_str_after_truncation = tokenizer.decode(context_tokens_after_truncation)
#     context_length_change_info = {
#         "original_context_str_length": len(prompt["context"]),
#         "context_str_length_after_truncation": len(context_str_after_truncation),
#         "original_context_token_length": len(context_tokens_before_truncation),
#         "context_token_length_after_truncation": len(context_tokens_after_truncation)
#     }
#     modified_prompt = create_prompt(question=prompt["question"], context=context_str_after_truncation, prompt_mode = prompt_mode)
#     return modified_prompt, context_length_change_info

def post_process_answers(answers):
    modified_answers = []
    for x in answers:
        if not x:
            modified_answers.append("")
        else:
            modified_answers.append(x.strip().split("\n")[0])
    return modified_answers

def num_gpt_tokens_per_message(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
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
        return num_gpt_tokens_per_message(messages, model="gpt-4-0613")
        # raise NotImplementedError(
        #     f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        # )
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
        # print("Warning: model not found. Using cl100k_base encoding.")
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
