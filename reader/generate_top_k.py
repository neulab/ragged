#file to generate the results of the QA taking retrival file as input

from readers.flanT5.flanT5_reader import FlanT5Reader
import argparse
import os
import time
import traceback

from tqdm import tqdm
from file_utils import save_jsonl, load_jsonl, save_json
from readers.llama2.llama2_reader import LlamaReader

time_map = {}


def post_process_answers(answers):
    return [x.strip().split("\n")[0] for x in answers]

def generate_reader_outputs(input_path, reader_object, output_file=None, start_offset=0, end_offset=None, top_k=1, args=None):
    
    retriever_data = load_jsonl(input_path)
    reader_responses = load_jsonl(output_file) if os.path.exists(output_file) else []
    print(f"no.of. questions in range {start_offset} to {end_offset} for which response is already generated = {len(reader_responses)}")

    error_file_path = output_file[:-6]+"_errors.jsonl"
    error_logs = load_jsonl(error_file_path, sort_by_id=False) if os.path.exists(error_file_path) else []

    reader_ques_ids_already_generated = [x['id'] for x in reader_responses] #can modify this to combined_jsonl file

    if not end_offset:
        end_offset = len(retriever_data)
    end_offset = min(end_offset, len(retriever_data))

    all_prompts = []
    prompt_indices = []
    time1 = time.time()
    for i, ques_info in tqdm(enumerate(retriever_data[start_offset:end_offset])):
        print("index : ", start_offset+i)

        if ques_info["id"] in reader_ques_ids_already_generated:
            continue

        question = ques_info["input"]+"?"

        if top_k:
            retrieved_passages = ques_info["output"][0]["provenance"][:top_k]
            context = "\n".join([passage["text"] for passage in retrieved_passages])
        else:
            context = ""
        
        prompt = {"question" : question, "context": context}
        all_prompts.append(prompt)
        prompt_indices.append(i)
            
    
    chunks = [list(zip(prompt_indices, all_prompts))[x:x+100] for x in range(0, len(all_prompts), 100)]
    all_answers = []
    all_context_length_changes = []
    for chunkid, chunk in enumerate(chunks):
        chunk_prompts = [prompt for _, prompt in chunk]
        try:
            answers, context_length_changes = reader_object.generate(chunk_prompts, max_new_tokens=10)
            all_context_length_changes.extend(context_length_changes)
            print(answers)
            answers = post_process_answers(answers)
            all_answers.extend(answers)
            chunk_prompt_indices = [x[0] for x in chunk]
            for q_index, answer in zip(chunk_prompt_indices, answers):
                ques_info = retriever_data[start_offset:end_offset][q_index]
                reader_responses.append({
                    "id" : ques_info["id"],
            "input" : ques_info["input"],
            "retrieved_passages": ques_info["output"][0]["provenance"][:top_k],
            "answer": answer
                })
                

        except Exception:
            print(f"Exception in {chunkid} chunk")
            print(traceback.format_exc())

            error_logs.append(
                {
                    "chunk_id" : chunkid,
                    "error": traceback.format_exc()
                }
            )
            save_jsonl(error_logs, error_file_path)
        save_jsonl(reader_responses, output_file)

    time2 = time.time()
    time_map["complete_generation"] = time2-time1
    print("Total reader_responses : ", len(reader_responses))
    print("Time taken: ", time_map["complete_generation"])
    save_jsonl(reader_responses, output_file)
    save_json(all_context_length_changes, f"/data/user_data/jhsia2/dbqa/reader_results/{args.model}/{args.dataset}/{args.retriever}/{'baseline' if args.top_k==0 else 'top'+str(args.top_k)}/reader_output_index_{args.start_offset}_to_{args.end_offset}_context_length_changes.json")
            

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted_api_path", type=str, default="babel-1-23:9426")
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--model", type=str)
    parser.add_argument("--retriever", type=str)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()

    model_class_dict = {
        "llama" : LlamaReader,
        "flanT5" : FlanT5Reader,
        "flanUl2" : FlanT5Reader,
        "llama_7b": LlamaReader
    }

    retriever_path_map = {
        "bm25": "/data/user_data/jhsia2/dbqa/retriever_results/predictions/bm25/",
        "colbert": "/data/user_data/jhsia2/dbqa/retriever_results/predictions/colbert/"

    }

    dataset_map = {
        "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
        "nq": "nq-dev-kilt.jsonl"
    }
    
    reader=model_class_dict[args.model](hosted_api_path =f"http://{args.hosted_api_path}:9426/")
    
    retriever_data_path = f"{retriever_path_map[args.retriever]}{dataset_map[args.dataset]}"

    # output_path = f"/data/user_data/afreens/kilt/{args.model}/{args.dataset}/{args.retriever}/top{args.top_k}/"
    output_path = f"/data/user_data/jhsia2/dbqa/reader_results/{args.model}/{args.dataset}/{args.retriever}/{'baseline' if args.top_k==0 else 'top'+str(args.top_k) }/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'{output_path}reader_output_index_{args.start_offset}_to_{args.end_offset}.jsonl'
    # output_file = f'{args.output_dir}reader_output_index_{args.start_offset}_to_{args.end_offset}.jsonl'

    
    generate_reader_outputs(retriever_data_path, reader, output_file=output_file, start_offset=args.start_offset, end_offset=args.end_offset, top_k=args.top_k, args=args)

    print("DONE!")