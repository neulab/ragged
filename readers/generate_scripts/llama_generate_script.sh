#!/bin/bash
#
#SBATCH --job-name=kilt_llama_generation
#SBATCH --output="/data/user_data/afreens/kilt/llama/nq/exp2/top50/logs/slurm-%A_%a.out"
#SBATCH --time=600


python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/llama/nq/exp2/top50/ --dataset nq --model llama2 --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k $4

# /data/user_data/afreens/kilt/llama/nq/exp2/top1/reader_output_index_0_to_3000.jsonl