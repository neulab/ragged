from transformers import LlamaTokenizer, T5Tokenizer

BASE_FOLDER = "path/to/base/folder"
DATA_FOLDER = f"{BASE_FOLDER}/data"
RETRIEVER_FOLDER = f"{BASE_FOLDER}/retriever_results"
READER_FOLDER = f"{BASE_FOLDER}/reader_results"

complete_model_names = {
   "llama2_7b": "huggingface/meta-llama/Llama-2-7b-hf",
    "llama2_70b": "huggingface/meta-llama/Llama-2-70b-hf",
    "flanT5": "huggingface/google/flan-t5-xxl",
    "flanUl2": "huggingface/google/flan-ul2"
}

dataset_map = {
    "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
    "nq": "nq-dev-kilt.jsonl",
    "bioasq": "bioasq.jsonl",
}

tokenizer_path_map = {
    "llama2_7b": "/data/datasets/models/meta-ai/llama2/weights/",
    "llama2_70b": "/data/datasets/models/meta-ai/llama2/weights/",
    "flanT5": "google/flan-ul2",
    "flanUl2": "google/flan-ul2"
}

tokenizer_map = {
    "llama2_7b": LlamaTokenizer,
    "llama2_70b": LlamaTokenizer,
    "flanT5": T5Tokenizer,
    "flanUl2": T5Tokenizer
}


def get_tokenizer(model_name):
    if model_name in tokenizer_map and model_name in tokenizer_path_map:
        return tokenizer_map[model_name].from_pretrained(tokenizer_path_map[model_name])
    else:
        raise Exception(f"{model_name} is not supported. Add the corresponding tokenizer and tokenizer_path in utils")
