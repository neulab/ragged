#!/bin/bash
#
#SBATCH --job-name=kilt_flanT5_generation
#SBATCH --output="/data/user_data/jhsia2/dbqa/reader_results/flanT5/hotpotqa/slurm-%A_%a.out"
#SBATCH --time=800


export PYTHONPATH=/home/afreens/ragged
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 0
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 1
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 2
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 3
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 5
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 10
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 20
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 30
python /home/afreens/ragged/readers/generate_top_k.py --retriever colbert --model flanT5 --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 50
