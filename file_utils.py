import json
import pdb
import os

def load_jsonl(filename, sort_by_id = True):
    print('loading from', filename)
    data = []
    with open(filename, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data

def save_jsonl(data, filename):
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok= True)
    print('writing to', filename)
    with open(filename, "w") as outfile:
        for idx, element in enumerate(data):
            json.dump(element, outfile)
            outfile.write("\n")

def save_json(data, filename):
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok= True)
    print('writing to', filename)
    # assert filename.endswith("json"), "file provided to save_json does not end with .json extension. Please recheck!"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)

def load_json(filename, sort_by_id = False):
    print('loading from', filename)
    assert filename.endswith("json"), "file provided to load_json does not end with .json extension. Please recheck!"
    data = json.load(open(filename))
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data
