import json
input_file = "/data/user_data/jhsia2/dbqa/data/kilt_knowledgesource.json"
output_file = "/data/user_data/jhsia2/dbqa/data/paragraph_knowledgesource.jsonl"
# total num paragraphs 111789997
# total num documents 5903530
num_par_docs = 0
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for l_id, line in enumerate(infile):
        if (l_id%100000==0):
            print(l_id)
        data = json.loads(line)
        num_par_docs += len(data['text'])
        for p_id, p in enumerate(data['text']):
            id = data['wikipedia_id'] + '_' + str(p_id+1)
            contents = p
            new_data = {
                "id": id,
                "contents": contents
            }
            outfile.write(json.dumps(new_data) + "\n")
print('done')
print('num_par_docs = ', num_par_docs)