import json
import csv
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, args.dataset + '.jsonl')
    output_file = os.path.join(args.data_dir, args.dataset + '-queries.tsv')

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