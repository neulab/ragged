#file to generate the results of the QA taking retrival file as input

from readers.flanT5.flanT5_reader import FlanT5Reader
import argparse
import os
import time
import traceback

from tqdm import tqdm
from file_utils import read_json, store_data, load_data, write_json
from readers.llama2.llama2_reader import LlamaReader

time_map = {}


def post_process_answers(answers):
    return [x.strip().split("\n")[0] for x in answers]

def generate_reader_outputs(input_path, reader_object, output_file=None, start_offset=0, end_offset=None):
    
    retriever_data = read_json(input_path)
    reader_responses = load_data(output_file) if os.path.exists(output_file) else []
    print(f"no.of. questions in range {start_offset} to {end_offset} for which response is already generated = {len(reader_responses)}")


    reader_ques_ids_already_generated = [x['id'] for x in reader_responses] #can modify this to combined_jsonl file

    if not end_offset:
        end_offset = len(retriever_data)
    end_offset = min(end_offset, len(retriever_data))

    
    prompt_indices = []
    time1 = time.time()
    for i, ques_info in tqdm(enumerate(retriever_data[start_offset:end_offset])):
        print("index : ", start_offset+i)
        all_prompts = []

        if ques_info["id"] in reader_ques_ids_already_generated:
            continue

        question = ques_info["input"]+"?"
        
        for context_info in ques_info["output"]:
            prompt = {"id": ques_info["id"], "question" : question, "context": context_info["retrieved"][0]["text"]}
            all_prompts.append(prompt)
            prompt_indices.append(i)
        answers, context_length_changes = reader_object.generate(all_prompts)
        for p, a in zip(all_prompts, answers):
            p["generated_answer"] = a
        reader_responses.append({
            "id" : ques_info["id"],
            "input" : ques_info["input"],
            "retrieved_passages" : all_prompts,
            "all_answers" : answers

        })

    store_data(output_file, reader_responses)
            

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted_api_path", type=str, default="babel-1-23:9426")
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=None)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    base_folder = "/data/user_data/jhsia2/dbqa/data/"
    args = get_args()

    model_class_dict = {
        "llama" : LlamaReader,
        "flanT5" : FlanT5Reader
    }

    dataset_map = {
        "hotpotqa" : "gold_hotpotqa_zeno_file.json",
        "nq": "gold_nq_zeno_file.json"
    }
    
    reader=model_class_dict[args.model](hosted_api_path =f"http://{args.hosted_api_path}:9426/")

    # output_path = f"/data/user_data/afreens/kilt/{args.model}/{args.dataset}/{args.retriever}/top{args.top_k}/"
    output_path = f"/data/user_data/jhsia2/dbqa/reader_results/{args.model}/{args.dataset}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'{output_path}gold_baseline_answers.jsonl'
    # output_file = f'{args.output_dir}reader_output_index_{args.start_offset}_to_{args.end_offset}.jsonl'
    base_file = f"{base_folder}{dataset_map[args.dataset]}"
    
    generate_reader_outputs(base_file, reader, output_file=output_file, start_offset=args.start_offset, end_offset=args.end_offset)

    print("DONE!")