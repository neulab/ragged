import json
import pdb

BASE_FOLDER = '/data/tir/projects/tir6/general/afreens/dbqa'
READER_BASE_FOLDER = '/data/tir/projects/tir6/general/afreens/dbqa/reader_results'
READER_BASE_FOLDER_NON_GOLD = "/data/tir/projects/tir6/general/afreens/dbqa/noisy_reader_results"

def load_jsonl(file_path, sort_by_id = True):
    print('loading from', file_path)
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data

def save_jsonl(data, filename):
    print('writing to', filename)
    with open(filename, "w") as outfile:
        for idx, element in enumerate(data):
            json.dump(element, outfile)
            outfile.write("\n")

def save_json(data, filename):
    print('writing to', filename)
    # assert filename.endswith("json"), "file provided to save_json does not end with .json extension. Please recheck!"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)

def load_json(filename, sort_by_id = False):
    print('reading from', filename)
    assert filename.endswith("json"), "file provided to load_json does not end with .json extension. Please recheck!"
    data = json.load(open(filename))
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data
