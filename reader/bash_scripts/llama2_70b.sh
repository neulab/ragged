#!/bin/bash
#SBATCH --job-name=llama2_70b_1
#SBATCH --exclude=shire-1-10,shire-1-6
#SBATCH --output=host_logs/llama2_70b_1.log
#SBATCH --error=host_logs/llama2_70b_1.err
#SBATCH --nodes=1
#SBATCH --exclude=babel-2-12,babel-9-3,babel-9-7,babel-9-11,babel-9-15,babel-9-19,babel-11-13,babel-8-7,babel-8-3,babel-11-17,babel-11-21,babel-3-11,babel-3-3,babel-3-32,babel-3-36,babel-4-13,babel-4-18,babel-6-11,babel-6-13,babel-6-15,babel-6-17,babel-6-5,babel-6-7,babel-6-9,babel-7-1,babel-7-17,babel-7-21,babel-7-25,babel-7-29,babel-7-37,babel-7-5,babel-7-9,shire-1-6
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=100GB
#SBATCH --time=10:00:00

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='meta-llama/Llama-2-70b-hf'

echo $model_id

mkdir -p /scratch/jhsia2
text-generation-launcher --model-id $model_id --port 9451 --master-port 23457 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096