


import csv
import json
import argparse
import os
import warnings
import pdb
from file_utils import load_json, save_json, load_jsonl


def convert_reader_results_to_zeno(reader_output_data, retriever_eval_data, is_bioasq = False):

    if is_bioasq:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_key = 'start_paragraph_id'
        section_name = 'par'

    # pdb.set_trace()
    # reader_output_data = sorted(reader_output_data, key=lambda x: x["id"])
    # retriever_eval_data = sorted(retriever_eval_data, key=lambda x: x["id"])

    print(len(reader_output_data), len(retriever_eval_data))
    assert len(reader_output_data) == len(retriever_eval_data)

    for i in range(len(reader_output_data)):
        if(reader_output_data[i]['id'] != retriever_eval_data[i]['id']):
            print(reader_output_data[i]['id'], retriever_eval_data[i]['id'])

       
    assert [d["id"] for d in reader_output_data] == [d["id"] for d in retriever_eval_data]

    zeno_format_data = []
    for reader_q_info, retriever_info in zip(reader_output_data, retriever_eval_data):
        assert reader_q_info["id"] == retriever_info["id"]
        qid = reader_q_info["id"]
        question = reader_q_info["input"]
        answer = reader_q_info["answer"]
        answer_evaluation = reader_q_info["answer_evaluation"]
        for retrieved_passage_info, retrieved_passage_info_eval in zip(reader_q_info["retrieved_passages"], retriever_info["doc-level results"][:len(reader_q_info["retrieved_passages"])]):
            # pdb.set_trace()
            # assert retrieved_passage_info["docid"] == retrieved_passage_info_eval["wiki_par_id"]
            retrieved_passage_info.update(retrieved_passage_info_eval)
            
        zeno_format_data.append({
            "id": qid,
            "input": question,
            "output":{
                "answer" : answer,
                "answer_evaluation": answer_evaluation,
                "retrieved" : reader_q_info["retrieved_passages"], 
                "summary context evaluation": {
                f"{docid_name}_id_match": any([r[f"{docid_name}_id_match"] for r in reader_q_info["retrieved_passages"]]),
                f"{docid_name}_{section_name}_id_match": any([r[f"{docid_name}_{section_name}_id_match"] for r in reader_q_info["retrieved_passages"]])
            }
            },
            "gold_answers": reader_q_info["gold_answers"],
        })
    assert (len(zeno_format_data) ==  len(reader_output_data))
    return zeno_format_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--retriever", help='retriever')
    parser.add_argument("--reader", help='reader')
    parser.add_argument("--dataset", help='dataset; nq-dev-kilt or hotpotqa-dev-kilt or bioasq')
    args = parser.parse_args()
    
    top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20", "top30", "top50"]
    # top_ks = ["top50"]
    # metrics_map = {}
    # metrics_save_path = "/data/user_data/afreens/kilt/llama/combined_metrics.json"
    # retriever_model = 'flan'
    # base_dir ='/data/tir/projects/tir6/general/afreens/dbqa'
    base_dir = '/data/tir/projects/tir6/general/afreens/dbqa'
    for top_k in top_ks:
        print(top_k)
        # k_dir = os.path.join(base_dir,'reader_results', args.reader, args.dataset, args.retriever, 'exp2', top_k)
        k_dir = os.path.join(base_dir,'reader_results', args.reader, args.dataset.split('-')[0], args.retriever, top_k)
        # base_folder = f"/data/user_data/afreens/kilt/flanT5/nq/exp2/{top_k}/"
        evaluation_file_path = os.path.join(k_dir, 'all_data_evaluated.jsonl')
        retriever_eval_file = os.path.join(base_dir, f'retriever_results/evaluations/{args.retriever}/{args.dataset}.jsonl')

        reader_output_data = load_jsonl(evaluation_file_path, sort_by_id = True)
        if (len(reader_output_data) == 0):
            warnings.warn("Warning: EMPTY file")
            continue
        retriever_eval_data = load_jsonl(retriever_eval_file, sort_by_id = True)
        if (len(retriever_eval_data) == 0):
            warnings.warn("Warning: EMPTY file")
            continue
        
        zeno_format_data = convert_reader_results_to_zeno(reader_output_data, retriever_eval_data, 'bioasq' in args.dataset)
        save_json(zeno_format_data, os.path.join(k_dir, "reader_results_zeno.json"))
