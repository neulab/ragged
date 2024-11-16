#!/bin/bash
#SBATCH --job-name=flanT5_1
#SBATCH --partition=general
#SBATCH --output=host_logs/flanT5_1.log
#SBATCH --error=host_logs/flanT5_1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --time=5:00:00

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='google/flan-t5-base'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="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"

mkdir -p /scratch/jhsia2
text-generation-launcher --model-id $model_id --port 6100 --master-port 23457 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048