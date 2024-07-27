from datasets import load_dataset
import json
import os
from datasets import Dataset, DatasetDict
from file_utils import load_jsonl, save_jsonl
from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np
import argparse
import pdb
def retrieve(query, index, model, top_k = 50):
    # Encode the query using the GTR model
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Perform the retrieval
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]


def get_retriever_outputs(model, index, corpus_dataset, query_dataset, top_k = 50):
    retrieved_output = []
    for qi, q in enumerate(query_dataset):
        print(qi)
        query = q['input']
        indices, distances = retrieve(query, index, model, top_k)
        provenance = []
        for (i,d) in zip(indices, distances):
            id = corpus_dataset[(int)(i)]['id']
            page_id = id.split('_')[0]
            start_par_id = id.split('_')[1]
            end_par_id = start_par_id
            content = corpus_dataset[(int)(i)]['contents']
            score = (float)(1/d)

            provenance.append({"page_id": page_id,
            "start_par_id": start_par_id,
            "end_par_id": end_par_id,
            "text": content,
            "score":score })
        
        retrieved_output.append({"id": q['id'], 
        "input": q['input'], 
        "output": [{"provenance": provenance}]})
    return retrieved_output
    


def main(args):
    retriever_name = 'gtr'

    # 2. Load the tokenizer and encoder
    print('Loading the question encoder and tokenizer')
    model = SentenceTransformer("sentence-transformers/gtr-t5-base")

    # 1. Load the corpus dataset
    print("Loading the corpus dataset")
    corpus_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files/{args.corpus}/{args.corpus}_jsonl/{args.corpus}.jsonl'
    corpus_dataset = load_dataset('json', data_files=corpus_file_name)
    if isinstance(corpus_dataset, DatasetDict):
        corpus_dataset = corpus_dataset['train']

    # passages = []
    # with open(corpus_file_name, 'r') as file:
    #     for i, line in enumerate(file):
    #         data = json.loads(line)
    #         passages.append({'id': data['id'], 'contents': data['contents']})
    #         if i == 50:
    #             break
    # corpus_dataset = Dataset.from_list(passages)

    # 3. Get corpus embeddings
    print('Getting corpus embeddings')
    index_file_name = os.path.join(args.index_dir, retriever_name, args.corpus + '.npy')
    if os.path.exists(index_file_name):
        print('Loading encoded corpus from', index_file_name)
        context_embeddings = np.load(index_file_name)
    else:
        passages = corpus_dataset['contents']
        context_embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
        context_embeddings = context_embeddings.astype(np.float32)
        os.makedirs(os.path.dirname(index_file_name), exist_ok=True)
        # print('Saving encoded corpus to', index_file_name)
        # np.save(index_file_name, context_embeddings)
    index = faiss.IndexFlatIP(context_embeddings.shape[1]) 
    index.add(context_embeddings)

    # 4. Load query dataset
    print('Loading the query dataset')
    query_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/{args.dataset}.jsonl'
    query_dataset = load_jsonl(query_file_name)

    # 5. Get retriever outputs
    print('Getting retriever outputs')
    retrieved_output = get_retriever_outputs(model, index, corpus_dataset, query_dataset, top_k = 50)
    save_jsonl(retrieved_output, os.path.join(args.prediction_dir, retriever_name, args.dataset + '.jsonl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--prediction_dir", help="retriever model name", default = '/data/tir/projects/tir6/general/afreens/dbqa/data/predictions')
    parser.add_argument("--index_dir", help="retriever model name", default = '/data/tir/projects/tir6/general/afreens/dbqa/indices')
    parser.add_argument("--corpus", help='datasetname')
    parser.add_argument("--dataset", help='datasetname')
    # parser.add_argument("--data_dir", help='datasetname')
    args = parser.parse_args()
    main(args)

    
