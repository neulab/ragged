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

def encode_passages(examples, context_tokenizer, context_encoder):
        inputs = context_tokenizer(examples['contents'], return_tensors='pt', truncation=True, padding=True, max_length = 512)
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output.cpu().numpy()
        return {'embeddings': embeddings}

def retrieve(query, index, question_tokenizer, question_encoder, top_k):
    # Encode the query using the question encoder
    inputs = question_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length = 512)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

    # Perform the retrieval
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]


def get_retriever_outputs(question_tokenizer, question_encoder, index, corpus_dataset, query_dataset, top_k):
    retrieved_output = []
    for qi, q in enumerate(query_dataset):
        if qi%100==0:
            print(qi, end = ' ')
        query = q['input']
        indices, distances = retrieve(query, index, question_tokenizer, question_encoder, top_k)
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
    retriever_name = 'dpr'
    # 2a. Load the question encoder and tokenizer
    print('Loading the question encoder and tokenizer')
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # 2b. Load the context encoder and tokenizer
    print('Loading the context encoder and tokenizer')
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # 1. Load the corpus dataset
    
    index_file_name = os.path.join(args.index_dir, retriever_name, args.corpus)
    faiss_index_path = os.path.join(args.index_dir, 'faiss_' + retriever_name, args.corpus)
    if os.path.exists(faiss_index_path):
        print('Loading Faiss index from', faiss_index_path)
        index = faiss.read_index(faiss_index_path)
    else:
        if os.path.exists(index_file_name):
            print('Loading encoded corpus from', index_file_name)
            encoded_corpus_dataset = Dataset.load_from_disk(index_file_name)
            if isinstance(encoded_corpus_dataset, DatasetDict):
                encoded_corpus_dataset = encoded_corpus_dataset['train']
            corpus_dataset = encoded_corpus_dataset
        else:
            print("Loading the corpus dataset")
            corpus_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files/{args.corpus}/{args.corpus}_jsonl/{args.corpus}.jsonl'
            corpus_dataset = load_dataset('json', data_files=corpus_file_name)

            # subset of datset
            # passages = []
            # with open(corpus_file_name, 'r') as file:
            #     for i, line in enumerate(file):
            #         data = json.loads(line)
            #         passages.append({'id': data['id'], 'contents': data['contents']})
            #         if i == 50:
            #             break
            # corpus_dataset = Dataset.from_list(passages)

            print('Getting corpus embeddings')
            encoded_corpus_dataset = corpus_dataset.map(encode_passages, batched=True, batch_size=2000, fn_kwargs={'context_tokenizer': context_tokenizer, 'context_encoder': context_encoder})
            os.makedirs(os.path.dirname(index_file_name), exist_ok=True)
            print('Saving encoded corpus to', index_file_name)
            encoded_corpus_dataset.save_to_disk(index_file_name)
        # Run the following line of code if you want to save the encoded corpus dataset
        # encoded_corpus_dataset.save_to_disk(index_file_name)
        print('Creating Faiss index')
        context_embeddings = np.vstack(encoded_corpus_dataset['embeddings']).astype(np.float32)
        index = faiss.IndexFlatIP(context_embeddings.shape[1]) 
        index.add(context_embeddings)
    
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(index, faiss_index_path)


    # 4. Load query dataset
    print('Loading the query dataset')
    query_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/{args.dataset}.jsonl'
    query_dataset = load_jsonl(query_file_name)

    # 5. Get retriever outputs
    print('Getting retriever outputs')
    retrieved_output = get_retriever_outputs(question_tokenizer, question_encoder, index, corpus_dataset, query_dataset, top_k = 50)
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

    
