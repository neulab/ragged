import argparse
import os
import time
import traceback

from tqdm import tqdm
from file_utils import save_jsonl, load_jsonl, save_json
from reader.reader_model import Reader
from reader.utils import merge_retriever_data_and_eval_results, post_process_answers
from utils import READER_FOLDER, RETRIEVER_FOLDER, get_tokenizer, dataset_map

time_map = {}

def generate_reader_outputs(retriever_data, reader_object, output_path=None, args=None):  
    if args.retriever == 'no_context':
        k = 0 
    elif args.retriever == 'gold':
        k = None
    else:
        k = args.k
    batch_size = args.batch_size
    output_file = os.path.join(output_path, 'reader_results.jsonl')
    additional_metadata_file = os.path.join(output_path, 'additional_metadata.jsonl')

    reader_responses = load_jsonl(output_file) if os.path.exists(output_file) else []
    print(f"no.of. questions in range for which response is already generated = {len(reader_responses)}")

    error_file_path = os.path.join(output_path, 'reader_errors.jsonl')
    error_logs = load_jsonl(error_file_path, sort_by_id=False) if os.path.exists(error_file_path) else []

    reader_ques_ids_already_generated = [x['id'] for x in reader_responses] #can modify this to combined_jsonl file
    all_prompts = []
    prompt_indices = []
    for i, ques_info in tqdm(enumerate(retriever_data)):
        if ques_info["id"] in reader_ques_ids_already_generated:
            continue
        question = ques_info["input"]+"?"
        context_documents = ques_info["output"][0]["provenance"]
        if args.top_negative:
            context_documents = [r for r in context_documents if r["page_par_id_match"]==False]
        elif args.top_positive:
            context_documents = [r for r in context_documents if r["page_par_id_match"]==True]
        if k != 0:
            retrieved_passages = context_documents[:k]
            context = "\n".join([passage["text"] for passage in retrieved_passages])
        else:
            context = ""
        
        prompt = {"question" : question, "context": context}
        all_prompts.append(prompt)
        prompt_indices.append(i)
            
    chunks = [list(zip(prompt_indices, all_prompts))[x:x+batch_size] for x in range(0, len(all_prompts), batch_size)]
    all_answers = []
    all_context_length_changes = []
    for chunkid, chunk in enumerate(chunks):
        print(f'{chunkid}/{len(chunks)}')
        chunk_prompts = [prompt for _, prompt in chunk]
        try:
            answers, context_length_changes = reader_object.generate(chunk_prompts, max_new_tokens=args.max_new_tokens, truncate=args.max_truncation)
            all_context_length_changes.extend(context_length_changes)
            # print(answers)
            answers = post_process_answers(answers)
            all_answers.extend(answers)
            chunk_prompt_indices = [x[0] for x in chunk]
            for q_index, answer in zip(chunk_prompt_indices, answers):
                ques_info = retriever_data[q_index]
                reader_responses.append({
                    "id" : ques_info["id"],
                    "input" : ques_info["input"],
                    "retrieved_passages": context_documents[:k],
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

    print("Total reader_responses : ", len(reader_responses))
    save_jsonl(reader_responses, output_file)
    save_json(all_context_length_changes, additional_metadata_file)
            

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted_api_endpoint", type=str, help="the hosted endpoint of the TGI model server. SHould be of the format - <node>:<port>")
    parser.add_argument("--k", type=int, default=1, help="the number of retrieved contexts to include in input for reader generation")
    parser.add_argument("--batch_size", type=int, default=50, help="the number of reader inputs processed simultaneously")
    parser.add_argument("--model_name", type=str, help="model name; <results_base_folder>/<model>")
    parser.add_argument("--retriever", type=str, help="retriever name; results stored at <results_base_folder>/<model>/<retriever>")
    parser.add_argument("--dataset", type=str, help="dataset name; results stored at <results_base_folder>/<model>/<retriever>/<dataset>")
    parser.add_argument("--max_new_tokens", type=int, help="number of tokens that the model would generate.")
    parser.add_argument("--max_truncation", type=int, default=4000, help="number of tokens fed to the reader model. If the input (i.e instruction+contexts+question) are greater than this value, they are truncated to these many tokens")
    parser.add_argument("--retrieval_mode", help = 'top_k, top_negative, top_positive, no_context, or gold?')
    # parser.add_argument("--top_positive", action='store_true', help="When this flag is set, contexts that are marked relevant are only filtered and fed to the reader model along with the instruction and question")
    # parser.add_argument("--top_negative", action='store_true', help="When this flag is set, contexts that are marked irrelevant are only filtered and fed to the reader model along with the instruction and question")

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()

    # define reader object
    tokenizer = get_tokenizer(args.model_name)
    reader=Reader(model_identifier=args.model_name, hosted_api_path =f"http://{args.hosted_api_endpoint}/", tokenizer=tokenizer)

    # get retriever data
    retriever_data_path = os.path.join(RETRIEVER_FOLDER, "predictions", args.retriever, dataset_map[args.dataset])
    retriever_eval_path = os.path.join(RETRIEVER_FOLDER, "evaluations", args.retriever, dataset_map[args.dataset])
    retriever_data = merge_retriever_data_and_eval_results(retriever_data_path, retriever_eval_path)
        
    final_model_name = f"{args.model_name}_{args.max_truncation}truncation_{args.max_new_tokens}new_tokens"
    
    output_path  = os.path.join(READER_FOLDER, args.final_model_name, args.dataset, args.retriever)
    
    if args.retriever != 'no_context' and args.retriever != 'gold':
        output_path = os.path.join(output_path, args.retrieval_mode, f'top_{args.k}')

    os.makedirs(output_path, exist_ok = True)
    
    generate_reader_outputs(retriever_data, reader, output_path=output_path, args=args)

    print("DONE!")