#!/bin/bash
#SBATCH --job-name=flanUL2
#SBATCH --output=host_logs/flanUL2_2.log
#SBATCH --error=host_logs/flanUL2_2.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=200GB
#SBATCH --time=0-12:00:00

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='google/flan-ul2'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="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"

mkdir -p /scratch/jhsia2

text-generation-launcher --model-id $model_id --port 9100 --master-port 23457 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048