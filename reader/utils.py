import os

from file_utils import load_jsonl


INSTRUCTION_STR = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_INSTRUCTION_STR = "Give simple short one phrase answers for the question"

def truncate_prompt(prompt, tokenizer, instruction_str_tokens, total_tokens, max_new_tokens):
    question_tokens = tokenizer(prompt["question"])["input_ids"]
    remaining_length = total_tokens-len(instruction_str_tokens)-len(question_tokens)-max_new_tokens-5 #additional buffer of 5
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
    return [x.strip().split("\n")[0] for x in answers]

def create_prompt(question, context):
    if context:
        return f"{INSTRUCTION_STR}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_INSTRUCTION_STR}\nQuestion: {question}\nAnswer: ".strip()


def merge_retriever_data_and_eval_results(retriever_data_path, retriever_eval_data_path):
    if os.path.exists(retriever_data_path) and os.path.exists(retriever_eval_data_path):
        retriever_data = load_jsonl(retriever_data_path)
        retriever_eval_data = load_jsonl(retriever_eval_data_path)
        for retriever_info, eval_info in zip(retriever_data, retriever_eval_data):
            for r,e in zip(retriever_info["output"][0]["provenance"], eval_info["page-level results"]):
                r["page_par_id_match"] = e["page_par_id_match"]
        return retriever_data
    elif os.path.exists(retriever_data_path):
        return load_jsonl(retriever_data_path)
    else:
        raise Exception(f"{retriever_data_path} does not exist!")
