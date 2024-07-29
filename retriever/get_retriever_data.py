from datasets import load_dataset
import json
import os
from datasets import Dataset, DatasetDict
from file_utils import load_jsonl, save_jsonl
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pdb

def retrieve(query, index, question_tokenizer, question_encoder, top_k, device):
    inputs = question_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]

def get_retriever_outputs(question_tokenizer, question_encoder, index, corpus_dataset, query_dataset, top_k, device, start_idx, end_idx):
    retrieved_output = []
    
    for qi in range(start_idx, end_idx):
        q = query_dataset[qi]
        if qi % 100 == 0:
            print(qi, end=' ')
        query = q['input']
        indices, distances = retrieve(query, index, question_tokenizer, question_encoder, top_k, device)
        provenance = []
        for (i, d) in zip(indices, distances):
            id = corpus_dataset[(int)(i)]['id']
            page_id = id.split('_')[0]
            start_par_id = id.split('_')[1]
            end_par_id = start_par_id
            content = corpus_dataset[(int)(i)]['contents']
            score = (float)(1 / d)
            provenance.append({
                "page_id": page_id,
                "start_par_id": start_par_id,
                "end_par_id": end_par_id,
                "text": content,
                "score": score
            })
        retrieved_output.append({
            "id": q['id'],
            "input": q['input'],
            "output": [{"provenance": provenance}]
        })
    return retrieved_output

def load_embeddings(embedding_path):
    return np.load(embedding_path)

def stack_all_embeddings(index_dir, retriever_name, corpus, num_batches):
    all_embeddings = []
    for batch_run in range(1, num_batches + 1):
        embeddings_filename = os.path.join(index_dir, f'{retriever_name}', f'{corpus}_{batch_run}.npy')
        if os.path.exists(embeddings_filename):
            print(f'Loading embeddings from {embeddings_filename}')
            embeddings = load_embeddings(embeddings_filename)
            all_embeddings.append(embeddings)
        else:
            print(f'Embeddings file {embeddings_filename} does not exist.')
    if all_embeddings:
        stacked_embeddings = np.vstack(all_embeddings).astype(np.float32)
        return stacked_embeddings
    else:
        print('No embeddings loaded.')
        return None

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever_name = 'dpr'
    
    # Load the question encoder and tokenizer
    print('Loading the question encoder and tokenizer')
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder = torch.nn.DataParallel(question_encoder)
    question_encoder.to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    

    # Faiss index path
    faiss_index_path = os.path.join(args.index_dir, f'faiss_{retriever_name}', args.corpus)
    stacked_embeddings = stack_all_embeddings(args.index_dir, retriever_name, args.corpus, args.num_batches)
    print(stacked_embeddings.size())
    if os.path.exists(faiss_index_path):
        # Load Faiss index if it exists
        print('Loading Faiss index from', faiss_index_path)
        index = faiss.read_index(faiss_index_path)
    else:
        # Stack all embeddings and create Faiss index if it does not exist
        # stacked_embeddings = stack_all_embeddings(args.index_dir, retriever_name, args.corpus, args.num_batches)
        # print(stacked_embeddings.size())
        
        if stacked_embeddings is not None:
            print('Creating Faiss index')
            index = faiss.IndexFlatIP(stacked_embeddings.shape[1])
            index.add(stacked_embeddings)
            os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
            faiss.write_index(index, faiss_index_path)
        else:
            print('No embeddings to create Faiss index.')
            return

    # Load corpus dataset
    print("Loading the corpus dataset")
    corpus_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files/{args.corpus}/{args.corpus}_jsonl/{args.corpus}.jsonl'
    corpus_dataset = load_dataset('json', data_files=corpus_file_name, cache_dir="/data/tir/projects/tir6/general/afreens/dbqa/cache")["train"]

    # Load query dataset
    print('Loading the query dataset')
    query_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/{args.dataset}.jsonl'
    query_dataset = load_jsonl(query_file_name)
    
    # Determine the start and end indices for this run
    total_queries = len(query_dataset)
    batch_size = args.batch_size
    start_idx = args.chunk * batch_size
    end_idx = min(start_idx + batch_size, total_queries)
    
    # Get retriever outputs for the specific chunk
    print(f'Getting retriever outputs for queries {start_idx} to {end_idx}')
    retrieved_output = get_retriever_outputs(question_tokenizer, question_encoder, index, corpus_dataset, query_dataset, top_k=50, device=device, start_idx=start_idx, end_idx=end_idx)
    output_filename = os.path.join(args.prediction_dir, retriever_name, f'{args.dataset}_chunk_{args.chunk}.jsonl')
    save_jsonl(retrieved_output, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--prediction_dir", help="retriever model name", default='/data/tir/projects/tir6/general/afreens/dbqa/data/predictions')
    parser.add_argument("--index_dir", help="retriever model name", default='/data/tir/projects/tir6/general/afreens/dbqa/indices')
    parser.add_argument("--corpus", help='dataset name', default="pubmed")
    parser.add_argument("--dataset", help='dataset name', default="complete_bioasq")
    parser.add_argument("--num_batches", help='number of batch runs', type=int, default=1)
    parser.add_argument("--chunk", help='which chunk of the query dataset to process', type=int, default=1)
    parser.add_argument("--batch_size", help='batch size for processing queries', type=int, default=1000)
    args = parser.parse_args()
    print("abc")
    main(args)
