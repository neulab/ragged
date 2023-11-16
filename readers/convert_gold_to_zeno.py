


import csv
import json
from kilt.kilt_utils import load_data
import argparse
import os
from file_utils import read_json, write_json

# def read_json(filename):
#     print('reading from ', filename)
#     with open(filename) as f:
#         data = json.load(f)
#     return data

# def write_json(data, filename):
#     print('writing into ', filename)
#     with open(filename, 'w') as f:
#         json.dump(data,f)


def convert_gold_to_zeno():
    # wiki_par_ids_data = read_json("/data/user_data/jhsia2/dbqa/data/gold-nq-dev-kilt.json")
    gold_data = load_data("/data/user_data/jhsia2/dbqa/data/nq-dev-kilt.jsonl")
    # tsv_file = "/data/user_data/jhsia2/dbqa/data/kilt_knowledgesource.tsv"

    par_id_to_text_map = {}

    with open("/data/user_data/jhsia2/dbqa/data/kilt_knowledgesource.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            print(i)
            par_id_to_text_map[row[0]] = row[1]
            # break
    zeno_format_data = []
    for ques_info in gold_data:
        qid = ques_info["id"]
        # print(qid)
        question = ques_info["input"]
        answers = ques_info["output"]
        output_answers = []
        for answer in answers:
            if "answer" not in answer or "provenance" not in answer:
                continue
            for p in answer["provenance"]:
                p["text"] = par_id_to_text_map[str(p["wikipedia_id"])+"_"+str(p["start_paragraph_id"]+1)]

            output_answers.append({
                "answer": answer["answer"],
                "retrieved": answer["provenance"]
            })
            
        zeno_format_data.append({
            "id": qid,
            "input": question,
            "output": output_answers
        })
    return zeno_format_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='dataset')
    args = parser.parse_args()
    data_dir = os.path.join('/data/user_data/jhsia2/dbqa/data')

    zeno_format_data = convert_gold_to_zeno()
    write_json(zeno_format_data, os.path.join(data_dir, f'gold_{args.dataset}_zeno_file.json'))
