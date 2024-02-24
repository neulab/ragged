#file to generate the results of the QA taking retrival file as input

from reader.flanT5.flan_reader import FlanReader
import argparse
import os
import time
import traceback

from tqdm import tqdm
from file_utils import BASE_FOLDER, READER_BASE_FOLDER, load_jsonl, save_jsonl
from reader.llama2.llama2_reader import LlamaReader

time_map = {}


def post_process_answers(answers):
    return [x.strip().split("\n")[0] for x in answers]

def generate_reader_outputs(input_path, reader_object, output_file=None, start_offset=0, end_offset=None, args=None):
    
    retriever_data = load_jsonl(input_path)
    reader_responses = load_jsonl(output_file) if os.path.exists(output_file) else []
    print(f"no.of. questions in range {start_offset} to {end_offset} for which response is already generated = {len(reader_responses)}")


    reader_ques_ids_already_generated = [x['id'] for x in reader_responses] #can modify this to combined_jsonl file

    if not end_offset:
        end_offset = len(retriever_data)
    end_offset = min(end_offset, len(retriever_data))

    all_prompts = []
    prompt_indices = []
    time1 = time.time()
    for i, ques_info in tqdm(enumerate(retriever_data[start_offset:end_offset])):
        # print("index : ", start_offset+i)

        if ques_info["id"] in reader_ques_ids_already_generated:
            continue

        question = ques_info["input"].strip("?")+"?"

        evidence_spans = []
        for o in ques_info["output"][0]['provenance']:
            evidence_spans.append(o["text"])
        # assert len(evidence_spans)>0
        evidence_span = "\n".join(evidence_spans)
        # ques_info["evidence_span"] = evidence_span
        prompt = {"id": ques_info["id"], "question" : question, "context": evidence_span}
        all_prompts.append(prompt)
        prompt_indices.append(i)
        
    chunks = [list(zip(prompt_indices, all_prompts))[x:x+100] for x in range(0, len(all_prompts), 100)]
    all_answers = []
    all_context_length_changes = []
    for chunkid, chunk in enumerate(chunks):
        chunk_prompts = [prompt for _, prompt in chunk]
        answers, context_length_changes = reader_object.generate(chunk_prompts, max_new_tokens=args.max_new_tokens, truncate=args.max_truncation)
        all_context_length_changes.extend(context_length_changes)
        # print(answers)
        answers = post_process_answers(answers)
        all_answers.extend(answers)
        chunk_prompt_indices = [x[0] for x in chunk]
        for q_index, answer in zip(chunk_prompt_indices, answers):
            ques_info = retriever_data[start_offset:end_offset][q_index]
            reader_responses.append({
                "id" : ques_info["id"],
        "input" : ques_info["input"],
        "retrieved_passages": ques_info["output"][0]["provenance"],
        "answer": answer
            })

        save_jsonl(reader_responses, output_file)
    save_jsonl(reader_responses, output_file)
            

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted_api_path", type=str, default="babel-1-23")
    parser.add_argument("--hosted_api_port", type=str, default="9426")
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=None)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--max_truncation", type=int, default=4000)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()

    model_class_dict = {
        "llama_70b" : LlamaReader,
        "flanT5" : FlanReader,
        "flanUl2" : FlanReader,
        "llama_7b": LlamaReader,
        "llama_70b_256_tokens": LlamaReader,
        "llama_70b_2000_truncation": LlamaReader,
        "llama_7b_256_tokens": LlamaReader,
        "llama_7b_2000_truncation" : LlamaReader
    }

    dataset_map = {
        "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
        "nq": "nq-dev-kilt.jsonl",
        "bioasq": "bioasq.jsonl",
        "complete_bioasq": "complete_bioasq.jsonl"
    }
    
    reader=model_class_dict[args.model](hosted_api_path =f"http://{args.hosted_api_path}:{args.hosted_api_port}/")

    # output_path = f"/data/user_data/afreens/kilt/{args.model}/{args.dataset}/{args.retriever}/top{args.top_k}/"
    output_path = f"{READER_BASE_FOLDER}/{args.model}/{args.dataset}/gold/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'{output_path}/gold/gold_baseline_answers.jsonl'
    gold_file = f"{BASE_FOLDER}/retriever_results/predictions/gold/{dataset_map[args.dataset]}"
    
    generate_reader_outputs(gold_file, reader, output_file=output_file, start_offset=args.start_offset, end_offset=args.end_offset, args=args)

    print("DONE!")