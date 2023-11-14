import glob
import os
from file_utils import load_data, store_data, write_json

CONTEXT_PROMPT = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_PROMPT = "Give simple short one phrase answers for the question"
def create_prompt(question, context):
    if context:
        return f"{CONTEXT_PROMPT}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_PROMPT}\nQuestion: {question}\nAnswer: ".strip()
    

def combine_all_files(base_path, output_path=None):
    all_data = []
    all_data_unique = []
    if os.path.exists(f"/{base_path.strip('/')}/all_data.jsonl"):
        return load_data(f"/{base_path.strip('/')}/all_data.jsonl")
    for file in glob.glob(f"/{base_path.strip('/')}/*"):
        if not file.endswith(".jsonl") or "error" in file:
            continue
        all_data.extend(load_data(file))

    qids = set()
    for x in all_data:
        if x["id"] in qids:
            continue
        qids.add(x["id"])
        all_data_unique.append(x)
    print(len(all_data_unique))
    # assert len(all_data_unique) == 2837
    
    if output_path:    
        store_data(output_path, all_data_unique)
    return all_data_unique




    

