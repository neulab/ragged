


import re
import argparse
import os
import warnings
from file_utils import load_json, save_json, load_jsonl


def convert_reader_results_to_zeno(reader_output_data, retriever_eval_data, is_baseline):
    assert len(reader_output_data) == len(retriever_eval_data), f'len of reader output ({len(reader_output_data)})!= len of retriever output ({len(retriever_eval_data)})'
    assert [d["id"] for d in reader_output_data] == [d["id"] for d in retriever_eval_data]

    zeno_format_data = []
    for reader_q_info, retriever_info in zip(reader_output_data, retriever_eval_data):
        assert reader_q_info["id"] == retriever_info["id"]
        qid = reader_q_info["id"]
        
        question = reader_q_info["input"]
        answer = reader_q_info["answer"]
        answer_evaluation = reader_q_info["answer_evaluation"]
        for retrieved_passage_info, retrieved_passage_info_eval in zip(reader_q_info["retrieved_passages"], retriever_info["passage-level results"][:len(reader_q_info["retrieved_passages"])]):
            # assert retrieved_passage_info["docid"] == retrieved_passage_info_eval["wiki_par_id"]
            retrieved_passage_info.update(retrieved_passage_info_eval)
        zeno_format_data.append({
            "id": qid,
            "input": question,
            "output":{
                "answer" : answer,
                "answer_evaluation": answer_evaluation,
                "retrieved" : reader_q_info["retrieved_passages"], 
                "summary context evaluation": 
                {
                f"page_id_match": False ,
                f"page_par_id_match": False,
                f"answer_in_context": False
                }
                if is_baseline else {
                f"page_id_match": False if len(reader_q_info['retrieved_passages'])== 0 else any([r[f"page_id_match"] for r in reader_q_info["retrieved_passages"]]),
                f"page_par_id_match": False if len(reader_q_info['retrieved_passages'])== 0 else any([r[f"page_par_id_match"] for r in reader_q_info["retrieved_passages"]]),
                f"answer_in_context": False if len(reader_q_info['retrieved_passages'])== 0 else any([r[f"answer_in_context"] for r in reader_q_info["retrieved_passages"]]),
            }
            },
            "gold_answers": reader_q_info["gold_answers"],
        })
    assert (len(zeno_format_data) ==  len(reader_output_data))
    return zeno_format_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--retriever", help='retriever or gold or no_context')
    parser.add_argument("--reader", help='reader')
    parser.add_argument("--retriever_evaluation_dir", help='reader')
    parser.add_argument("--reader_output_dir", help='reader')
    parser.add_argument("--retrieval_mode", help = 'top_k, top_negative, or top_positive?')
    parser.add_argument("--k_list", help = 'what are the comma separate list of k values?')
    parser.add_argument("--dataset", help='dataset; nq-dev-kilt or hotpotqa-dev-kilt or bioasq')
    args = parser.parse_args()
    # if your're considering the top negative or top positive documents within the top_k documents, then there is no gold or baseline.

    k_dir = os.path.join(args.reader_output_dir, args.reader, args.dataset, args.retriever)
    if args.retriever == 'no_context':
        # k_dir 
        retriever_eval_data = None

    elif args.retriever == 'gold':
        # k_dir = os.path.join(args.reader_output_dir, args.reader, args.dataset, args.retriever)
        retriever_eval_file = f'{args.retriever_evaluation_dir}/gold/{args.dataset}.jsonl'
        retriever_eval_data = load_jsonl(retriever_eval_file, sort_by_id = True)

    else:
        k_list_str = re.split(r',\s*', args.k_list)
        k_list = [int(num) for num in k_list_str]
        for k in k_list:
            print(k)
            k_dir = os.path.join(k_dir, args.retrieval_mode, f'top_{k}')
            retriever_eval_file = f'{args.retriever_evaluation_dir}/{args.retriever}/{args.dataset}.jsonl'
            retriever_eval_data = load_jsonl(retriever_eval_file, sort_by_id = True)
  
        evaluation_file_path = os.path.join(k_dir, 'all_data_evaluated.jsonl')
        
        reader_output_data = load_jsonl(evaluation_file_path, sort_by_id = True)
        # if (len(reader_output_data) == 0):
        #     warnings.warn("Warning: EMPTY file")
        #     continue

        
        # if (len(retriever_eval_data) == 0):
        #     warnings.warn("Warning: EMPTY file")
        #     continue
        
        zeno_format_data = convert_reader_results_to_zeno(reader_output_data, retriever_eval_data, args.retriever == 'no_context')

            
        save_json(zeno_format_data, os.path.join(k_dir, "compiled_results.json"))
