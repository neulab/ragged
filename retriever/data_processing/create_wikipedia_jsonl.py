import json
import argparse
import os
from file_utils import save_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--corpus_dir", help="retriever model name")
    args = parser.parse_args()

    input_file = os.path.join(args.corpus_dir, 'kilt_knowledgesource.json')
    output_file = os.path.join(args.corpus_dir, "wikipedia", "wikipedia_jsonl", "wikipedia.jsonl")
    os.makedirs(os.path.join(args.corpus_dir, "wikipedia", "wikipedia_jsonl"), exist_ok=True)

    num_par_docs = 0
    id2title = {}
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for l_id, line in enumerate(infile):
            if (l_id%100000==0):
                print(l_id)
            data = json.loads(line)
            id2title[data['wikipedia_id']] = data['wikipedia_title']
            num_par_docs += len(data['text'])
            for p_id, p in enumerate(data['text']):
                id = data['wikipedia_id'] + '_' + str(p_id)
                contents = p
                new_data = {
                    "id": id,
                    "contents": contents
                }
                outfile.write(json.dumps(new_data) + "\n")

    save_json(id2title, os.path.join(args.corpus_dir,'wikipedia', 'id2title.json'))
    print('done')
    print('num_par_docs = ', num_par_docs)