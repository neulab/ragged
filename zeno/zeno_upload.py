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

def create_project(dataset):
    if 'bioasq' in dataset:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_key = 'start_paragraph_id'
        section_name = 'par'

    project = client.create_project(
        name=f"Document QA (gold) - {dataset}",
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
                    # "gold context set": {"type": "text", "label": "gold text set: "},
                    "answer": {"type": "text", "label": "reader answer: "},
                    "retrieved context": {
                        "type": "list",
                        "elements": {
                            "type": "vstack",
                            "keys": {
                                # "score": {"type": "text", "label": "score: "},
                                f"{docid_name}_id": {"type": "markdown"},
                                "text": {"type": "text", "label": "text: "},
                                f"answer_in_context": {"type": "text", "label": f"answer_in_context: "},
                                f"{docid_name}_id_match": {"type": "text", "label": f"{docid_name}_id match: "},
                                f"{docid_name}_{section_name}_id_match": {"type": "text", "label": f"{docid_name}_{section_name}_id match: "}
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
            ZenoMetric(name="exact_match", type="mean", columns=["exact_match"]),
            ZenoMetric(name="f1", type="mean", columns=["f1"]),
            ZenoMetric(name="answer_in_context", type="mean", columns=["answer_in_context"]),
            # ZenoMetric(name="bertscore_precision", type="mean", columns=["bertscore_precision"]),
            # ZenoMetric(name="bertscore_recall", type="mean", columns=["bertscore_recall"]),
            # ZenoMetric(name="bertscore_f1", type="mean", columns=["bertscore_f1"]),
            # ZenoMetric(name=f"gold {docid_name}_{section_name}_id set size", type="mean", columns=[f"gold {docid_name}_{section_name}_id set size"]),
            ZenoMetric(name="substring_match", type="mean", columns=["substring_match"]),
            # ZenoMetric(name="any {docid_name}_id_match", type="mean", columns=["any {docid_name}_id_match"]),
            # ZenoMetric(name="any {docid_name}_{section_name}_id_match", type="mean", columns=["any {docid_name}_{section_name}_id_match"]),
            ZenoMetric(name=f"precision - {docid_name}_id_match", type="mean", columns=[f"precision {docid_name}_id_match"]),
            ZenoMetric(name=f"precision - {docid_name}_{section_name}_id_match", type="mean", columns=[f"precision {docid_name}_{section_name}_id_match"]),
            ZenoMetric(name=f"recall - {docid_name}_id_match", type="mean", columns=[f"recall {docid_name}_id_match"]),
            ZenoMetric(name=f"recall - {docid_name}_{section_name}_id_match", type="mean", columns=[f"recall {docid_name}_{section_name}_id_match"]),
        ],
    )
    return project
    

def get_hist_info(size_set, unit):
    print('avg', np.mean(size_set), 'std', np.std(size_set), 'min', np.min(size_set), 'max', np.max(size_set))

    plt.hist(size_set, bins=np.arange(min(size_set)-0.5, max(size_set)+1.5, 1), edgecolor='black')

    # Set the labels and title for the plot
    plt.xlabel(f'gold {unit} set size')
    plt.ylabel('frequency')
    plt.title(f'gold {unit} set size - {dataset}')

    # Set x-ticks to correspond to the integer values
    plt.xticks(range(min(size_set), max(size_set) + 1))

    # Display the plot
    plt.show()

def get_precision(guess_id_set, gold_id_set):
    precision = np.mean([[s in gold_id_set] for s in guess_id_set])
    return precision
def get_recall(guess_id_set, gold_id_set):
    recall = np.mean([[s in guess_id_set] for s in gold_id_set]) if len(gold_id_set) > 0 else 0.0
    return recall

def get_reader_df(top_k, combined_data, is_bioasq = False):
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
    return pd.DataFrame(
        {
            "question": [d['input'] for d in combined_data],
            # "dataset": [d['dataset'] for d in combined_data],
            "question_category": [d['question_category'] for d in combined_data],
            "id": [d['id'] for d in combined_data],
            "output": [
                json.dumps(
                    {   
                        "gold answer set": ', '.join(d['gold_answer_set']),
                        # "gold title set": ', '.join(d['gold_title_set']),
                        # "gold context set": '\n'.join(d['gold_text_set']),
                        # "gold context": d['gold_context'],
                        "answer": d["output"]["answer"],
                        "retrieved context": [
                            {
                                # "{docid_name}_id": None,
                                # "text": None,
                                # "score": None,
                                # "{docid_name}_id_match": None,
                                # "{docid_name}_{section_name}_id_match": None
                            }
                        ] if top_k == 'baseline' else [
                            {
                                f"{docid_name}_id": "[{idx}]({url})".format(
                                    idx=id2title.get(r[f"{docid_name}_id"], 'Title not available.'),
                                    url="https://pubmed.ncbi.nlm.nih.gov/"
                                    + r[f"{docid_name}_id"],
                                ),
                                "text": r["text"],
                                # "score": r.get('score', None),
                                f"{docid_name}_id_match": r[f"{docid_name}_id_match"],
                                f"{docid_name}_{section_name}_id_match": r[f"{docid_name}_{section_name}_id_match"],
                                "answer_in_context": r["answer_in_context"]
                            }
                            if len(r['text'])!=0 else {}
                        for r in d["output"]["retrieved"]
                        ],
                    }
                )
                for d in combined_data
            ],
            
            f"gold {docid_name}_{section_name}_id set size": [
                len(d[f"gold_{docid_name}_{section_name}_id_set"]) for d in combined_data
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
            # "bertscore_precision": [
            #     d["output"]["answer_evaluation"]["bertscore"]["bertscore_precision"] for d in combined_data
            # ],
            # "bertscore_recall": [
            #     d["output"]["answer_evaluation"]["bertscore"]["bertscore_recall"] for d in combined_data
            # ],
            # "bertscore_f1": [
            #     d["output"]["answer_evaluation"]["bertscore"]["bertscore_f1"] for d in combined_data
            # ],
            "exact_match": [
                d["output"]["answer_evaluation"]["exact_match"] for d in combined_data
            ],
            "substring_match": [
                d["output"]["answer_evaluation"]["substring_match"] for d in combined_data
            ],
            "answer_in_context": [
                d["output"]["summary context evaluation"]["answer_in_context"] for d in combined_data
            ],
            # f"any {docid_name}_id_match": [False for d in combined_data] if top_k == 'baseline' else [
            #     d["output"]["summary context evaluation"][f"{docid_name}_id_match"] for d in combined_data
            # ],
            # f"any {docid_name}_{section_name}_id_match": [False for d in combined_data] if top_k == 'baseline' else [
            #     d["output"]["summary context evaluation"][f"{docid_name}_{section_name}_id_match"] for d in combined_data
            # ],
            f"precision {docid_name}_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_precision(set([r.get(f"{docid_name}_id", None) for r in d["output"]["retrieved"]]), d[f'gold_{docid_name}_id_set']) for d in combined_data
            ],
            f"precision {docid_name}_{section_name}_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_precision(set([r.get(f"{docid_name}_{section_name}_id", None) for r in d["output"]["retrieved"]]), d[f'gold_{docid_name}_{section_name}_id_set']) for d in combined_data
            ],
            f"recall {docid_name}_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_recall(set([r.get(f"{docid_name}_id", None) for r in d["output"]["retrieved"]]), d[f'gold_{docid_name}_id_set']) for d in combined_data
            ],
            f"recall {docid_name}_{section_name}_id_match": [0 for d in combined_data] if top_k == 'baseline' else [
                get_recall(set([r.get(f"{docid_name}_{section_name}_id", None) for r in d["output"]["retrieved"]]), d[f'gold_{docid_name}_{section_name}_id_set']) for d in combined_data
            ],
        }
    )

def combine_gold_and_compiled(output_data, gold_data, questions_categorized, is_bioasq = False):
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
    
    for i, (od, gd) in enumerate(zip(output_data, gold_data)):
        if(od['id'] != gd['id']):
            print(od, gd)
            break
        # od['dataset'] = gd['dataset']
        od['question_category'] = questions_categorized[od['id']]
        od['gold_answer_set'] = gd['output']['answer_set']
        od[f'gold_{docid_name}_id_set'] = gd['output'][f'{docid_name}_id_set']
        od[f'gold_{docid_name}_{section_name}_id_set'] = gd['output'][f'{docid_name}_{section_name}_id_set']
        od['gold_title_set'] = gd['output']['title_set']
    return output_data

def combine_truncated_stats(combined_data, truncated_reader_stats):
    for (c,t) in zip(combined_data, truncated_reader_stats):
        c['truncated_num_docs'] = t['num_docs']
    return combined_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='dataset')
    parser.add_argument("--create_project", action='store_true', help='Flag to indicate project creation')
    args = parser.parse_args()
    load_dotenv(override=True)

    # root_dir = '/data/tir/projects/tir6/general/afreens/dbqa'
    root_dir = '/data/tir/projects/tir6/general/afreens/dbqa'
    results_dir = os.path.join(root_dir, 'reader_results')
    client = ZenoClient('zen_EZ7LuqItWgObcQmIvNZVytvhtTh8JMs2HrSzzfXsiIg')
    # print('hi')
    if 'bioasq' in args.dataset:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_key = 'start_paragraph_id'
        section_name = 'par'
    
    
    id2title = load_json(os.path.join(root_dir, f'data/corpus_files/{docid_name}_{section_name}_id2title.json'))
    # pdb.set_trace()

    dataset = args.dataset
    project = create_project(dataset)

    gold_data = load_json(os.path.join(root_dir, 'data/gold_zeno_files', f"gold_{dataset.split('complete_')[-1]}_zeno_file.json"), sort_by_id = True)

    # for d in gold_data:
    #     d['dataset'] = dataset

    # {docid_name}_{section_name}_id_set_size = []
    # {docid_name}_id_set_size = []
    # for d in gold_data:
    #     {docid_name}_{section_name}_id_set_size.append(len(d['output']['{docid_name}_{section_name}_id_set']))
    #     {docid_name}_id_set_size.append(len(d['output']['{docid_name}_id_set']))
    # get_hist_info({docid_name}_{section_name}_id_set_size, unit = 'paragraph')
    # get_hist_info({docid_name}_id_set_size, unit = 'page')
    
    questions_categorized = load_json(os.path.join(root_dir, f'data/questions_categorized/{dataset}_questions_categorized.json'))

    if args.create_project:
        data_df = pd.DataFrame({"question": [d["input"] for d in gold_data], 'id': [d['id'] for d in gold_data]})
        project.upload_dataset(data_df, id_column="id", data_column="question")

    reader_models = ['flanUl2', 'llama_70b', 'flanT5', 'llama_7b', 'llama_70b_256_tokens']
    reader_models = ['llama_70b_2000_truncation']
    # retriever_models = ['gold','colbert', 'bm25']
    retriever_models = ['colbert','gold']
    # retriever_models = ['gold']
    top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20", "top30", "top50"]
    # top_ks =["top30", "top50"]
    print(retriever_models)
    print(reader_models)
    print(top_ks)
    for retriever_model in retriever_models:
        for reader_model in reader_models:
        
            print('retriever', retriever_model)
            print('reader', reader_model)
            if retriever_model == 'gold':

                data = load_json(os.path.join(results_dir, reader_model, 'bioasq' if 'bioasq' in dataset else dataset, 'gold', "reader_results_zeno.json"))
                        
                combined_data = combine_gold_and_compiled(data, gold_data, questions_categorized,'bioasq' in args.dataset)
                output_df = get_reader_df('gold', combined_data, 'bioasq' in args.dataset)
                project.upload_system(
                    output_df, name= (reader_model + ' gold'), id_column="id", output_column="output"
                )
            else:
                for top_k in top_ks:
                    print(top_k)
                    data = load_json(os.path.join(results_dir, reader_model, dataset, retriever_model, f"{top_k}/reader_results_zeno.json"))
                            
                    combined_data = combine_gold_and_compiled(data, gold_data, questions_categorized,'bioasq' in args.dataset)
                    output_df = get_reader_df(top_k, combined_data, 'bioasq' in args.dataset)
                    if top_k == 'baseline':
                        project.upload_system(
                        output_df, name= (reader_model + ' ' + top_k), id_column="id", output_column="output"
                    )
                    else:
                        project.upload_system(
                            output_df, name= (retriever_model + ' ' + reader_model + ' ' + top_k), id_column="id", output_column="output"
                        )