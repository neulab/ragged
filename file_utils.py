import json
import pdb

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

def load_json(filename):
    print('reading from', filename)
    assert filename.endswith("json"), "file provided to load_json does not end with .json extension. Please recheck!"
    return json.load(open(filename))
