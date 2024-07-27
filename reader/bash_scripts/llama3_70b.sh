#!/bin/bash
#SBATCH --job-name=llama3_70b
#SBATCH --output=host_logs/llama3_70b_1.log
#SBATCH --error=host_logs/llama3_70b_1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='meta-llama/Meta-Llama-3-70B'

echo $model_id

mkdir -p /scratch/jhsia2
text-generation-launcher --model-id $model_id --port 8101 --master-port 23456 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 8000 --max-total-tokens 8192 --max-batch-prefill-tokens 8192