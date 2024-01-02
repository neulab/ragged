import json
import csv
import pdb
import os

root_dir = '/data/tir/projects/tir6/general/afreens/dbqa/data'
input_file = os.path.join(root_dir, "bioasq/complete_medline_corpus_jsonl/complete_medline_corpus.jsonl")
output_file = os.path.join(root_dir, "bioasq/complete_medline_corpus.tsv")

print('creating medline wiki tsv in ', output_file)
# print('total num documents = 5903530')
# num_par_docs = 0

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # tsv_writer = csv.writer(outfile, delimiter='\t')
    for l_id, line in enumerate(infile):
        # print(l_id)
        if (l_id%100000==0):
            print(l_id)
        data = json.loads(line)
        # for p_id, p in enumerate(data['text']):
        id = data['id']
        p = data['contents']
        outfile.write('\t'.join([id, p.encode('unicode_escape').decode()]) + '\n')
print('done')