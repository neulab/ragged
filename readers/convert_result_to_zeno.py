


import csv
import json
from kilt.kilt_utils import load_data
import argparse
import os
import pdb
# , read_json, write_json

def read_json(filename):
    print('reading from ', filename)
    with open(filename) as f:
        data = json.load(f)
    return data

def write_json(data, filename):
    print('writing into ', filename)
    with open(filename, 'w') as f:
        json.dump(data,f)

def convert_reader_results_to_zeno(reader_output_data, retriever_eval_data):
    # pdb.set_trace()
    reader_output_data = sorted(reader_output_data, key=lambda x: x["id"])
    retriever_eval_data = sorted(retriever_eval_data, key=lambda x: x["id"])

    assert [d["id"] for d in reader_output_data] == [d["id"] for d in retriever_eval_data]

    zeno_format_data = []
    for reader_q_info, retriever_info in zip(reader_output_data, retriever_eval_data):
        assert reader_q_info["id"] == retriever_info["id"]
        qid = reader_q_info["id"]
        question = reader_q_info["input"]
        answer = reader_q_info["answer"]
        answer_evaluation = reader_q_info["answer_evaluation"]
        for retrieved_passage_info, retrieved_passage_info_eval in zip(reader_q_info["retrieved_passages"], retriever_info["doc-level results"][:len(reader_q_info["retrieved_passages"])]):
            assert retrieved_passage_info["docid"] == retrieved_passage_info_eval["wiki_par_id"]
            retrieved_passage_info.update(retrieved_passage_info_eval)
            
        zeno_format_data.append({
            "id": qid,
            "input": question,
            "output":{
                "answer" : answer,
                "answer_evaluation": answer_evaluation,
                "retrieved" : reader_q_info["retrieved_passages"], 
                "summary context evaluation": {
                "wiki_id_match": any([r["wiki_id_match"] for r in reader_q_info["retrieved_passages"]]),
                "wiki_par_id_match": any([r["wiki_par_id_match"] for r in reader_q_info["retrieved_passages"]])
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
    parser.add_argument("--dataset", help='dataset')
    args = parser.parse_args()
    top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20", "top30", "top50"]
    # metrics_map = {}
    # metrics_save_path = "/data/user_data/afreens/kilt/llama/combined_metrics.json"
    # retriever_model = 'flan'
    base_dir = os.path.join('/data/user_data/jhsia2/dbqa')
    for top_k in top_ks:
        print(top_k)
        k_dir = os.path.join(base_dir,'reader_results', args.reader, args.dataset, args.retriever, 'exp2', top_k)
        # base_folder = f"/data/user_data/afreens/kilt/flanT5/nq/exp2/{top_k}/"
        evaluation_file_path = os.path.join(k_dir, 'all_data_evaluated.jsonl')
        retriever_eval_file = os.path.join(base_dir, f'retriever_results/evaluations/{args.retriever}/{args.dataset}-dev-kilt.jsonl')

        reader_output_data = load_data(evaluation_file_path)
        retriever_eval_data = load_data(retriever_eval_file)
        
        zeno_format_data = convert_reader_results_to_zeno(reader_output_data, retriever_eval_data)
        write_json(zeno_format_data, os.path.join(k_dir, "reader_results_zeno.json"))
