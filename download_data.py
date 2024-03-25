
from datasets import load_dataset
import json


# SPECIFY YOUR DATA_DIR AND CORPUS_DIR
data_dir = None
corpus_dir = None


# 1a. load and save corpus data as jsonl file
corpus_name = 'kilt_wikipedia'
dataset = load_dataset("jenhsia/ragged", corpus_name)
dataset['train'].to_json(f'{corpus_dir}/{corpus_name}/{corpus_name}_jsonl/{corpus_name}.jsonl')


# 1b. load and save corpus id2title as json file
dataset_name = f'{corpus_name}_id2title'
dataset = load_dataset("jenhsia/ragged", dataset_name)

id_title_dict = {}

for example in dataset:
    id_title_dict[example['id']] = example['title']

json_file_path = f'{corpus_dir}/{corpus_name}/id2title.json'

with open(json_file_path, 'w') as json_file:
    json.dump(id_title_dict, json_file)

# 2. load and save question dataset
dataset_name = 'nq'
dataset = load_dataset("jenhsia/ragged", dataset_name)
dataset['train'].to_json(f'{data_dir}/{dataset_name}.jsonl')