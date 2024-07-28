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
def encode_passages(examples, context_tokenizer, context_encoder, device):
    inputs = context_tokenizer(examples['contents'], return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = context_encoder(**inputs).pooler_output.cpu().numpy()
    return embeddings
def retrieve(query, index, question_tokenizer, question_encoder, top_k, device):
    inputs = question_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]
def get_retriever_outputs(question_tokenizer, question_encoder, index, corpus_dataset, query_dataset, top_k, device):
    retrieved_output = []
    for qi, q in enumerate(query_dataset):
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
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_run = args.batch_run
    retriever_name = 'dpr'
    print('Loading the question encoder and tokenizer')
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder = torch.nn.DataParallel(question_encoder)
    question_encoder.to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    print('Loading the context encoder and tokenizer')
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_encoder = torch.nn.DataParallel(context_encoder)
    context_encoder.to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    index_file_name = os.path.join(args.index_dir, f'{retriever_name}_{batch_run}', args.corpus)
    # print("index file", index_file_name)
    # if os.path.exists(index_file_name):
    #     print('Loading encoded corpus from', index_file_name)
    #     encoded_corpus_dataset = Dataset.load_from_disk(index_file_name)
    #     if isinstance(encoded_corpus_dataset, DatasetDict):
    #         encoded_corpus_dataset = encoded_corpus_dataset['train']
    #     corpus_dataset = encoded_corpus_dataset
    # else:
    print("Loading the corpus dataset")
    corpus_file_name = f'/data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files/{args.corpus}/{args.corpus}_jsonl/{args.corpus}.jsonl'
    corpus_dataset = load_dataset('json', data_files=corpus_file_name)["train"]
    print('Getting corpus embeddings in batches on GPU')
    batch_size = 8000
    encoded_batches = []
    print(corpus_dataset)
    for i in tqdm(range(int(1e7*(int(batch_run)-1)), int(1e7*int(batch_run)), batch_size), desc="Encoding Corpus"):
        print(i)
        # print(len(corpus_dataset[:100000]))
        batch = corpus_dataset[i:i + batch_size]
        # print(batch, len(batch))
        # pdb.set_trace()
        embeddings = encode_passages(batch, context_tokenizer, context_encoder, device)
        encoded_batches.append(embeddings)
        # pdb.set_trace()
        # break
    context_embeddings = np.vstack(encoded_batches).astype(np.float32)
    embeddings_filename = os.path.join(args.index_dir, f'{retriever_name}', f'{args.corpus}_{batch_run}.npy')
    os.makedirs(os.path.dirname(embeddings_filename), exist_ok=True)
    np.save(embeddings_filename, context_embeddings)
        # pdb.set_trace()
        # encoded_corpus_dataset = Dataset.from_dict({'embeddings': context_embeddings})
        
    print('Saving encoded corpus to', embeddings_filename)
    # encoded_corpus_dataset.save_to_disk(index_file_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--prediction_dir", help="retriever model name", default='/data/tir/projects/tir6/general/afreens/dbqa/data/predictions')
    parser.add_argument("--index_dir", help="retriever model name", default='/data/tir/projects/tir6/general/afreens/dbqa/indices')
    parser.add_argument("--corpus", help='datasetname', default="kilt_wikipedia")
    parser.add_argument("--dataset", help='datasetname', default="nq-dev-kilt")
    parser.add_argument("--batch_run")
    args = parser.parse_args()
    main(args)