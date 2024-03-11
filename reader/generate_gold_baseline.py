import argparse
import os
import time
import traceback

from tqdm import tqdm
from file_utils import load_jsonl, save_jsonl
from reader.reader_model import Reader
from reader.utils import post_process_answers
from utils import READER_FOLDER, RETRIEVER_FOLDER, get_tokenizer, dataset_map

time_map = {}

def generate_reader_outputs(retriever_data, reader_object, output_file=None, start_offset=0, end_offset=None, args=None):
    batch_size = args.batch_size
    reader_responses = load_jsonl(output_file) if os.path.exists(output_file) else []
    print(f"no.of. questions for which response is already generated = {len(reader_responses)}")

    reader_ques_ids_already_generated = [x['id'] for x in reader_responses] #can modify this to combined_jsonl file
    all_prompts = []
    prompt_indices = []
    time1 = time.time()
    for i, ques_info in tqdm(enumerate(retriever_data)):
        if ques_info["id"] in reader_ques_ids_already_generated:
            continue
        question = ques_info["input"].strip("?")+"?"
        evidence_spans = []
        for o in ques_info["output"][0]['provenance']:
            evidence_spans.append(o["text"])
        evidence_span = "\n".join(evidence_spans)
        prompt = {"id": ques_info["id"], "question" : question, "context": evidence_span}
        all_prompts.append(prompt)
        prompt_indices.append(i)
        
    chunks = [list(zip(prompt_indices, all_prompts))[x:x+100] for x in range(0, len(all_prompts), batch_size)]
    all_answers = []
    all_context_length_changes = []
    for _, chunk in enumerate(chunks):
        chunk_prompts = [prompt for _, prompt in chunk]
        answers, context_length_changes = reader_object.generate(chunk_prompts, max_new_tokens=args.max_new_tokens, truncate=args.max_truncation)
        all_context_length_changes.extend(context_length_changes)
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
    parser.add_argument("--hosted_api_endpoint", type=str, help="the hosted endpoint of the TGI model server. SHould be of the format - <node>:<port>")
    parser.add_argument("--model_name", type=str, help="model name; <results_base_folder>/<model>")
    parser.add_argument("--dataset", type=str, help="dataset name; results stored at <results_base_folder>/<model>/gold/<dataset>")
    parser.add_argument("--max_new_tokens", type=int, help="number of tokens that the model would generate.")
    parser.add_argument("--max_truncation", type=int, default=4000, help="number of tokens fed to the reader model. If the input (i.e instruction+contexts+question) are greater than this value, they are truncated to these many tokens")

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    
    # define reader object
    tokenizer = get_tokenizer(args.model_name)
    reader=Reader(hosted_api_path =f"http://{args.hosted_api_endpoint}/", tokenizer=tokenizer)
    final_model_name = f"{args.model_name}_{args.max_truncation}truncation_{args.max_new_tokens}new_tokens"
    output_path = os.path.join(READER_FOLDER, args.final_model_name, args.dataset, args.retriever, 'gold')
    # output_path = os.path.join(READER_FOLDER, args.model, args.dataset, "gold")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, 'reader_results.jsonl')
    gold_dataset_file = os.path.join(RETRIEVER_FOLDER, "predictions", "gold", dataset_map[args.dataset])
    gold_dataset = load_jsonl(gold_dataset_file)
    
    generate_reader_outputs(gold_dataset, reader, output_file=output_file, args=args)

    print("DONE!")