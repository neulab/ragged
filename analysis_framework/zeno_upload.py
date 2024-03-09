from zeno_client import ZenoClient, ZenoMetric
from file_utils import load_json
import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import re

def create_project(dataset):

    project = client.create_project(
        name=f"Document QA - {dataset}",
        view={
            "data": {"type": "text", 
                    "label": "question:"
                    },
            "label": {"type": "text"},
            "output": {
                "type": "vstack",
                "keys": {
                    "gold answer set": {"type": "text", "label": "gold answer set: "},
                    # "gold title set": {"type": "text", "label": "gold title set: "},
                    "answer": {"type": "text", "label": "reader answer: "},
                    "retrieved context": {
                        "type": "list",
                        "elements": {
                            "type": "vstack",
                            "keys": {
                                # "score": {"type": "text", "label": "score: "},
                                f"page_id": {"type": "markdown"},
                                "text": {"type": "text", "label": "text: "},
                                f"answer_in_context": {"type": "text", "label": f"answer_in_context: "},
                                f"page_id_match": {"type": "text", "label": f"page_id match: "},
                                f"page_par_id_match": {"type": "text", "label": f"page_par_id match: "}
                            },
                        },
                        "collapsible": "bottom",
                        "border": True,
                        "pad": True,
                    },
                },
            },
        },
        description="Document-grounded question answering with Wikipedia",
        metrics=[
            # ZenoMetric(name="max retrieved score", type="mean", columns=["max_score"]),
            # ZenoMetric(name="avg retrieved score", type="mean", columns=["avg_score"]),
            # ZenoMetric(name=f"gold page_par_id set size", type="mean", columns=[f"gold page_par_id set size"]),
            # ZenoMetric(name="any page_id_match", type="mean", columns=["any page_id_match"]),
            # ZenoMetric(name="any page_par_id_match", type="mean", columns=["any page_par_id_match"]),
            ZenoMetric(name="exact_match", type="mean", columns=["exact_match"]),
            ZenoMetric(name="f1", type="mean", columns=["f1"]),
            ZenoMetric(name="answer_in_context", type="mean", columns=["answer_in_context"]),
            ZenoMetric(name="substring_match", type="mean", columns=["substring_match"]),
            ZenoMetric(name=f"precision - page_id_match", type="mean", columns=[f"precision page_id_match"]),
            ZenoMetric(name=f"precision - page_par_id_match", type="mean", columns=[f"precision page_par_id_match"]),
            ZenoMetric(name=f"recall - page_id_match", type="mean", columns=[f"recall page_id_match"]),
            ZenoMetric(name=f"recall - page_par_id_match", type="mean", columns=[f"recall page_par_id_match"]),
        ],
    )
    return project

def get_precision(guess_id_set, gold_id_set):
    precision = np.mean([[s in gold_id_set] for s in guess_id_set])
    return precision
def get_recall(guess_id_set, gold_id_set):
    recall = np.mean([[s in guess_id_set] for s in gold_id_set]) if len(gold_id_set) > 0 else 0.0
    return recall

def get_reader_df(top_k, combined_data):
    return pd.DataFrame(
        {
            "question": [d['input'] for d in combined_data],
            "question_category": [d['question_category'] for d in combined_data],
            "id": [d['id'] for d in combined_data],
            "output": [
                json.dumps(
                    {   
                        "gold answer set": ', '.join(d['gold_answer_set']),
                        # "gold title set": ', '.join(d['gold_title_set']),
                        "answer": d["output"]["answer"],
                        "retrieved context": [
                            {}
                        ] if top_k == 'baseline' else [
                            {
                                f"page_id": "[{idx}]({url})".format(
                                    idx=id2title.get(r[f"page_id"], 'Title not available.'),
                                    url="https://pubmed.ncbi.nlm.nih.gov/"
                                    + r[f"page_id"],
                                ),
                                "text": r["text"],
                                # "score": r.get('score', None),
                                f"page_id_match": r[f"page_id_match"],
                                f"page_par_id_match": r[f"page_par_id_match"],
                                "answer_in_context": r["answer_in_context"]
                            }
                            if len(r['text'])!=0 else {}
                        for r in d["output"]["retrieved"]
                        ],
                    }
                )
                for d in combined_data
            ],
            
            f"gold page_par_id set size": [
                len(d[f"gold_page_par_id_set"]) for d in combined_data
            ],
            # "max_score": [0 for d in combined_data] if top_k == 'baseline' else [
            #     d["output"]["retrieved"][0]["score"] for d in combined_data
            # ],
            # "avg_score": [0 for d in combined_data] if top_k == 'baseline' else [
            #     np.mean([r["score"] for r in d["output"]["retrieved"]]) for d in combined_data
            # ],
            "f1": [
                d["output"]["answer_evaluation"]["f1"] for d in combined_data
            ],
            "exact_match": [
                d["output"]["answer_evaluation"]["exact_match"] for d in combined_data
            ],
            "substring_match": [
                d["output"]["answer_evaluation"]["substring_match"] for d in combined_data
            ],
            "answer_in_context": [
                d["output"]["summary context evaluation"]["answer_in_context"] for d in combined_data
            ],
            f"any page_id_match": [False for d in combined_data] if top_k == 'baseline' else [
                d["output"]["summary context evaluation"][f"page_id_match"] for d in combined_data
            ],
            f"any page_par_id_match": [False for d in combined_data] if top_k == 'baseline' else [
                d["output"]["summary context evaluation"][f"page_par_id_match"] for d in combined_data
            ],
            f"precision page_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_precision(set([r.get(f"page_id", None) for r in d["output"]["retrieved"]]), d[f'gold_page_id_set']) for d in combined_data
            ],
            f"precision page_par_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_precision(set([r.get(f"page_par_id", None) for r in d["output"]["retrieved"]]), d[f'gold_page_par_id_set']) for d in combined_data
            ],
            f"recall page_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_recall(set([r.get(f"page_id", None) for r in d["output"]["retrieved"]]), d[f'gold_page_id_set']) for d in combined_data
            ],
            f"recall page_par_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_recall(set([r.get(f"page_par_id", None) for r in d["output"]["retrieved"]]), d[f'gold_page_par_id_set']) for d in combined_data
            ],
        }
    )

def combine_gold_and_compiled(output_data, gold_data, questions_categorized):
    for i, (od, gd) in enumerate(zip(output_data, gold_data)):
        if(od['id'] != gd['id']):
            print(od, gd)
            break
        od['question_category'] = questions_categorized[od['id']]
        od['gold_answer_set'] = gd['output']['answer_set']
        od[f'gold_page_id_set'] = gd['output'][f'page_id_set']
        od[f'gold_page_par_id_set'] = gd['output'][f'page_par_id_set']
        od['gold_title_set'] = gd['output']['title_set']
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='dataset')
    parser.add_argument("--zeno_api_key")
    parser.add_argument("--reader_results_dir", help='reader_results_dir')
    parser.add_argument("--corpus_dir", help='reader_results_dir')
    parser.add_argument("--corpus", help='reader_results_dir')
    parser.add_argument("--dataset_dir", help='reader_results_dir')
    parser.add_argument("--top_ks", help = 'what are the comma separate list of k values?')
    parser.add_argument("--create_project", action='store_true', help='this overwrites past results and creates new project instead of resuming')
    args = parser.parse_args()
    load_dotenv(override=True)

    client = ZenoClient(args.zeno_api_key)
    
    
    id2title = load_json(os.path.join(args.corpus_dir, args.corpus, 'id2title.json'))

    project = create_project(args.dataset)

    gold_data = load_json(os.path.join(args.dataset_dir, 'gold_zeno_files', f"gold_{args.dataset}_zeno_file.json"), sort_by_id = True)

    
    questions_categorized = load_json(os.path.join(args.dataset_dir, f'{args.dataset}_questions_categorized.json'))

    if args.create_project:
        data_df = pd.DataFrame({"question": [d["input"] for d in gold_data], 'id': [d['id'] for d in gold_data]})
        project.upload_dataset(data_df, id_column="id", data_column="question")

    reader_models = re.split(r',\s*', args.reader_models)
    retriever_models = re.split(r',\s*', args.retriever_models)
    top_ks = terms_list = re.split(r',\s*', args.top_ks)

    print(retriever_models)
    print(reader_models)
    print(top_ks)
    for retriever_model in retriever_models:
        for reader_model in reader_models:
        
            print('retriever', retriever_model)
            print('reader', reader_model)
            if retriever_model == 'gold':

                data = load_json(os.path.join(args.reader_results_dir, reader_model, args.dataset, 'gold', "reader_results_zeno.json"))
                        
                combined_data = combine_gold_and_compiled(data, gold_data, questions_categorized)
                output_df = get_reader_df('gold', combined_data)
                project.upload_system(
                    output_df, name= (reader_model + ' gold'), id_column="id", output_column="output"
                )
            else:
                for top_k in top_ks:
                    print(top_k)
                    data = load_json(os.path.join(args.reader_results_dir, reader_model, args.dataset, retriever_model, f"{top_k}/reader_results_zeno.json"))
                            
                    combined_data = combine_gold_and_compiled(data, gold_data, questions_categorized)
                    output_df = get_reader_df(top_k, combined_data)
                    if top_k == 'baseline':
                        project.upload_system(
                        output_df, name= (reader_model + ' ' + top_k), id_column="id", output_column="output"
                    )
                    else:
                        project.upload_system(
                            output_df, name= (retriever_model + ' ' + reader_model + ' ' + top_k), id_column="id", output_column="output"
                        )