import json
import csv
import pdb
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--corpus")
    parser.add_argument("--corpus_dir")
    args = parser.parse_args()

    input_file = os.path.join(args.corpus_dir, f"{args.corpus}/{args.corpus}_jsonl/{args.corpus}.jsonl")
    output_file = os.path.join(args.corpus_dir, f"{args.corpus}/{args.corpus}.tsv")

    # print('creating medlin tsv in ', output_file)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for l_id, line in enumerate(infile):
            if (l_id%100000==0):
                print(l_id)
            data = json.loads(line)
            id = data['id']
            p = data['contents']
            outfile.write('\t'.join([id, p.encode('unicode_escape').decode()]) + '\n')
    
    print(f'writing to {output_file}')