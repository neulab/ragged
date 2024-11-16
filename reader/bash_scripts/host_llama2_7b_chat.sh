#!/bin/bash
#SBATCH --job-name=llama2_chat_7b_1
#SBATCH --partition=general
#SBATCH --output=host_logs/llama2_chat_7b_1.log
#SBATCH --error=host_logs/llama2_chat_7b_1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --time=10:00:00

echo $SLURM_JOB_ID

source /home/jhsia2/.bashrc
conda activate tgi-env

model_id='meta-llama/Llama-2-7b-chat-hf'

echo $model_id

mkdir -p /scratch/jhsia2
text-generation-launcher --model-id $model_id --port 1100 --master-port 23456 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096