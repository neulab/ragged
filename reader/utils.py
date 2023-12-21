import glob
import os
from file_utils import load_jsonl, save_jsonl
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers import LlamaTokenizer

CONTEXT_PROMPT = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_PROMPT = "Give simple short one phrase answers for the question"
def create_prompt(question, context):
    if context:
        return f"{CONTEXT_PROMPT}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_PROMPT}\nQuestion: {question}\nAnswer: ".strip()
    

def combine_all_files(base_path, output_path=None):
    all_data = []
    all_data_unique = []
    if os.path.exists(f"/{base_path.strip('/')}/all_data.jsonl"):
        data = load_jsonl(f"/{base_path.strip('/')}/all_data.jsonl")
        if len(data) > 0:
            return data
    for file in glob.glob(f"/{base_path.strip('/')}/*"):
        if not file.endswith(".jsonl") or "error" in file:
            continue
        all_data.extend(load_data(file))

    qids = set()
    for x in all_data:
        if x["id"] in qids:
            continue
        qids.add(x["id"])
        all_data_unique.append(x)
    print(len(all_data_unique))
    # assert len(all_data_unique) == 2837
    
    if output_path:    
        save_jsonl(output_path, all_data_unique)
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
    t5_context_prompt_tokenized = t5_tokenizer(CONTEXT_PROMPT)["input_ids"]
    llama_context_prompt_tokenized = llama_tokenizer(CONTEXT_PROMPT)["input_ids"]

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
    
        save_jsonl(write_file, retriever_data)
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

    t5_context_prompt_tokenized = t5_tokenizer(CONTEXT_PROMPT)["input_ids"]
    llama_context_prompt_tokenized = llama_tokenizer(CONTEXT_PROMPT)["input_ids"]

    for r_dp in tqdm(retriever_data):
        question = r_dp["input"]
        retrieved_passages = r_dp["output"][0]["provenance"]
        contexts = [passage["text"] for passage in retrieved_passages]
        r_dp["llama_truncation"] = get_contexts_truncation_index(llama_tokenizer, question, contexts, 4000, llama_context_prompt_tokenized)
        r_dp["t5_truncation"] = get_contexts_truncation_index(t5_tokenizer, question, contexts, 2000, t5_context_prompt_tokenized)

    save_jsonl(write_file, retriever_data)

if __name__ == "__main__":
    retriever_path_map = {
        "bm25": "/data/user_data/jhsia2/dbqa/retriever_results/predictions/bm25/",
        "colbert": "/data/user_data/jhsia2/dbqa/retriever_results/predictions/colbert/"

    }

    dataset_map = {
        "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
        "nq": "nq-dev-kilt.jsonl"
    }

    save_map = {
        "hotpotqa" : "hotpotqa-truncation.jsonl",
        "nq": "nq-truncation.jsonl"
    }
    find_tokenization_limits_based_on_contexts(f"{retriever_path_map['bm25']}{dataset_map['hotpotqa']}", f"{retriever_path_map['bm25']}{save_map['hotpotqa']}")
    find_tokenization_limits_based_on_contexts(f"{retriever_path_map['bm25']}{dataset_map['nq']}", f"{retriever_path_map['bm25']}{save_map['nq']}")
    find_tokenization_limits_based_on_contexts(f"{retriever_path_map['colbert']}{dataset_map['hotpotqa']}", f"{retriever_path_map['colbert']}{save_map['hotpotqa']}")
    find_tokenization_limits_based_on_contexts(f"{retriever_path_map['colbert']}{dataset_map['nq']}", f"{retriever_path_map['colbert']}{save_map['nq']}")

    # find_tokenization_limits(f"{retriever_path_map['bm25']}{dataset_map['hotpotqa']}", f"{retriever_path_map['bm25']}{save_map['hotpotqa']}")
    # find_tokenization_limits(f"{retriever_path_map['bm25']}{dataset_map['nq']}", f"{retriever_path_map['bm25']}{save_map['nq']}")
    # find_tokenization_limits(f"{retriever_path_map['colbert']}{dataset_map['hotpotqa']}", f"{retriever_path_map['colbert']}{save_map['hotpotqa']}")
    # find_tokenization_limits(f"{retriever_path_map['colbert']}{dataset_map['nq']}", f"{retriever_path_map['colbert']}{save_map['nq']}")
