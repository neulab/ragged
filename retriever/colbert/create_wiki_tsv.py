import json
import csv
import pdb
input_file = "/data/tir/projects/tir6/general/afreens/dbqa/data/kilt_knowledgesource.json"
output_file = "/data/tir/projects/tir6/general/afreens/dbqa/data/kilt_knowledgesource.tsv"

print('creating kilt wiki tsv in ', output_file)
print('total num documents = 5903530')
# num_par_docs = 0

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # tsv_writer = csv.writer(outfile, delimiter='\t')
    for l_id, line in enumerate(infile):
        if (l_id%100000==0):
            print(l_id)
        data = json.loads(line)
        for p_id, p in enumerate(data['text']):
            id = data['wikipedia_id'] + '_' + str(p_id+1)
            outfile.write('\t'.join([id, p]))
print('done')