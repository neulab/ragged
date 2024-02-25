import glob
import os
from file_utils import BASE_FOLDER, READER_BASE_FOLDER, load_json, load_jsonl, save_jsonl
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers import LlamaTokenizer

INSTRUCTION_STR = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_INSTRUCTION_STR = "Give simple short one phrase answers for the question"

def truncate_prompt(prompt, tokenizer, instruction_str_tokens, total_tokens):
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

def create_prompt(question, context):
    if context:
        return f"{INSTRUCTION_STR}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_INSTRUCTION_STR}\nQuestion: {question}\nAnswer: ".strip()
    
def combine_all_files(base_path, output_path=None):
    all_data = []
    all_data_unique = []
    if os.path.exists(os.path.join(base_path, "all_data.jsonl")):
        data = load_jsonl(os.path.join(base_path, "all_data.jsonl"))
        if len(data) > 0:
            return data
    for file in glob.glob(os.path.join(base_path, "*")):
        if not file.endswith(".jsonl") or "error" in file:
            continue
        all_data.extend(load_jsonl(file))

    qids = set()
    for x in all_data:
        if x["id"] in qids:
            continue
        qids.add(x["id"])
        all_data_unique.append(x)
    print(len(all_data_unique))
    if output_path:    
        save_jsonl(all_data_unique, output_path)
    return all_data_unique

def find_tokenization_limits(retriever_output_file, write_file):

    def get_diff(tokenizer, question, context, total_tokens, context_prompt_tokenized):
        question_tokenized = tokenizer(question)["input_ids"]
        remaining_length = total_tokens-len(context_prompt_tokenized)-len(question_tokenized)-10
        context_tokenized_without_truncation = tokenizer(context, add_special_tokens=False)
        context_tokenized = tokenizer(context, max_length=remaining_length, truncation=True, add_special_tokens=False)
        return len(context_tokenized_without_truncation["input_ids"]) - len(context_tokenized["input_ids"])

        
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    llama_tokenizer = LlamaTokenizer.from_pretrained("/data/datasets/models/meta-ai/llama2/weights/")
    retriever_data = load_data(retriever_output_file)

    top_ks = list(range(20, 51, 5))
    t5_context_prompt_tokenized = t5_tokenizer(INSTRUCTION_STR)["input_ids"]
    llama_context_prompt_tokenized = llama_tokenizer(INSTRUCTION_STR)["input_ids"]

    for r_dp in tqdm(retriever_data):
        r_dp["t5_truncation"] = {}
        r_dp["llama_truncation"] = {}

    llama_truncated_passages_count_per_k = {}
    t5_truncated_passages_count_per_k = {}
    for top_k in top_ks:
        llama_truncated_passages_count = 0
        t5_truncated_passages_count = 0
        for r_dp in tqdm(retriever_data):
            question = r_dp["input"]
            retrieved_passages = r_dp["output"][0]["provenance"][:top_k]
            context = "\n".join([passage["text"] for passage in retrieved_passages])
            t5_tokens_diff = get_diff(t5_tokenizer, question, context, 2000, t5_context_prompt_tokenized)
            llama_tokens_diff = get_diff(llama_tokenizer, question, context, 4000, llama_context_prompt_tokenized)
            if t5_tokens_diff > 10:
                t5_truncated_passages_count+=1
                r_dp["t5_truncation"][top_k] = t5_tokens_diff
            if llama_tokens_diff > 10:
                llama_truncated_passages_count+=1
                r_dp["llama_truncation"][top_k] = llama_tokens_diff
        llama_truncated_passages_count_per_k[top_k] = (llama_truncated_passages_count, float(llama_truncated_passages_count/len(retriever_data)))
        t5_truncated_passages_count_per_k[top_k] = (t5_truncated_passages_count, float(t5_truncated_passages_count/len(retriever_data)))
    
        save_jsonl(retriever_data, write_file)
        print(retriever_output_file)
        print("LLAMA: ", llama_truncated_passages_count_per_k)
        print("T5: ", t5_truncated_passages_count_per_k)

def find_tokenization_limits_based_on_contexts(retriever_output_file, write_file):

    def get_contexts_truncation_index(tokenizer, question, contexts, total_tokens, context_prompt_tokenized):
        question_tokenized = tokenizer(question)["input_ids"]
        remaining_length = total_tokens-len(context_prompt_tokenized)-len(question_tokenized)-10
        i = 0
        for i, context in enumerate(contexts):
            context_tokenized = tokenizer(context, add_special_tokens=False)["input_ids"]
            remaining_length-=len(context_tokenized)
            if remaining_length <= 0:
                return i
        return 50 if remaining_length>0 else i

        
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    llama_tokenizer = LlamaTokenizer.from_pretrained("/data/datasets/models/meta-ai/llama2/weights/")
    retriever_data = load_data(retriever_output_file)

    t5_context_prompt_tokenized = t5_tokenizer(INSTRUCTION_STR)["input_ids"]
    llama_context_prompt_tokenized = llama_tokenizer(INSTRUCTION_STR)["input_ids"]

    for r_dp in tqdm(retriever_data):
        question = r_dp["input"]
        retrieved_passages = r_dp["output"][0]["provenance"]
        contexts = [passage["text"] for passage in retrieved_passages]
        r_dp["llama_truncation"] = get_contexts_truncation_index(llama_tokenizer, question, contexts, 4000, llama_context_prompt_tokenized)
        r_dp["t5_truncation"] = get_contexts_truncation_index(t5_tokenizer, question, contexts, 2000, t5_context_prompt_tokenized)

    save_jsonl(retriever_data, write_file)

def convert_gold_files():
    models = "llama_70b_2000_truncation llama_70b_256_tokens llama_7b_2000_truncation llama_7b_256_tokens".split(" ")
    datasets = "nq hotpotqa bioasq".split(" ")
    for model in models:
        for dataset in datasets:
            input_file = f"{READER_BASE_FOLDER}/{model}/{dataset}/gold/all_data_evaluated.jsonl"
            if os.path.exists(input_file):
                data = load_jsonl(input_file)
                for dp in data:
                    dp["retrieved_passages"] = [{"text":dp["evidence_span"]}]
                save_jsonl(data, input_file)

def add_question_type_to_bioasq():
    dataset_map = {
        "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
        "nq": "nq-dev-kilt.jsonl",
        "bioasq": "bioasq.jsonl",
        "complete_bioasq": "complete_bioasq.jsonl"
    }

    type_file = "/data/tir/projects/tir6/general/afreens/dbqa/data/questions_categorized/bioasq_questions_categorized.json"
    type_data = load_json(type_file)
    for dataset in "bioasq complete_bioasq".split(" "):
        gold_file = f"{BASE_FOLDER}/data/{dataset_map[dataset]}"
        gold_data = load_jsonl(gold_file)
        for dp in gold_data:
            dp["question_type"] = type_data[dp["id"]]

        save_jsonl(gold_data, gold_file)

def merge_retriever_data_and_eval_results(retriever_data, retriever_eval_data):
    for retriever_info, eval_info in zip(retriever_data, retriever_eval_data):
        for r,e in zip(retriever_info["output"][0]["provenance"], eval_info["page-level results"]):
            r["page_par_id_match"] = e["page_par_id_match"]
    return retriever_data
