import json
import csv
import argparse
import os

# bioasq 3837 queries with exact answers
# corpus 207413

def main(dataset):
    data_dir = "/data/user_data/jhsia2/dbqa/data/"
    input_file = os.path.join(data_dir, dataset + '.jsonl')
    output_file = os.path.join(data_dir, dataset + '-queries.tsv')
    print('creating data tsv in ', output_file)
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for l_id, line in enumerate(infile):
            if (l_id%1000==0):
                print(l_id)
            data = json.loads(line)
            id = data['id']
            contents = data['input']
            outfile.write('\t'.join([id, contents.encode('unicode_escape').decode()]) + '\n')
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='datasetname')
    args = parser.parse_args()
    main(args.dataset)