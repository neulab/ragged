#!/bin/bash
#
#SBATCH --job-name=kilt_flanT5_generation
#SBATCH --output="/data/user_data/afreens/kilt/flanT5/hotpot/slurm-%A_%a.out"
#SBATCH --time=600

python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top1/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 1
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top2/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 2
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top3/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 3
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top5/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 5
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top10/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 10
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top20/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 20
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top30/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 30
python /home/afreens/KILT/kilt/readers/generate_top_k.py --output_dir /data/user_data/afreens/kilt/flanT5/hotpot/exp2/top50/ --model flanT5 --dataset hotpot --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 50
