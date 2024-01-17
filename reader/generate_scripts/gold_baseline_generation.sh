#!/bin/bash
#
#SBATCH --job-name=gold_baseline
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/gold_gen/slurm-%A_%a.out"
#SBATCH --time=800


export PYTHONPATH=/home/afreens/ragged
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_7b_2000_truncation --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path babel-1-23 --hosted_api_port 8102 --max_new_tokens 10 --max_truncation 2000

# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_256_tokens --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path babel-1-31 --hosted_api_port 8103 --max_new_tokens 256 --max_truncation 4000
python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_256_tokens --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path babel-1-31 --hosted_api_port 7103 --max_new_tokens 256 --max_truncation 4000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_7b_256_tokens --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path babel-0-37 --hosted_api_port 8102 --max_new_tokens 256 --max_truncation 4000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_7b_256_tokens --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path babel-0-37 --hosted_api_port 7102 --max_new_tokens 256 --max_truncation 4000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_256_tokens --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path babel-4-36 --hosted_api_port 9429 --max_new_tokens 256 --max_truncation 4000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_256_tokens --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path babel-1-31 --hosted_api_port 8103 --max_new_tokens 256 --max_truncation 4000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_2000_truncation --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path babel-4-36 --hosted_api_port 9429 --max_new_tokens 10 --max_truncation 2000
# python /home/afreens/ragged/reader/generate_gold_baseline.py --model llama_70b_2000_truncation --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path babel-1-31 --hosted_api_port 8103 --max_new_tokens 10 --max_truncation 2000


