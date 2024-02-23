#!/bin/bash
#SBATCH --job-name=2flanUL2
#SBATCH --partition=general
#SBATCH --output=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/2flanUL2.out
#SBATCH --error=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/2flanUL2.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=150GB
#SBATCH --time=1-00:00:00
#SBATCH --exclude=shire-1-10,shire-1-6,babel-5-11,babel-5-15

export user=jhsia2

mkdir -p /scratch/$user/tgi-uds-socket-1

source /home/jhsia2/.bashrc
conda activate tgi-env

OPENSSL_DIR=`pwd`/.openssl \
OPENSSL_LIB_DIR=`pwd`/.openssl/lib \
OPENSSL_INCLUDE_DIR=`pwd`/.openssl/include \
BUILD_EXTENSIONS=false \
    make install-server

echo $SLURM_JOB_ID

model_id='google/flan-ul2'

echo $model_id



export HUGGING_FACE_HUB_TOKEN="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"

text-generation-launcher --model-id $model_id --port 9400 --master-port 23457 --shard-uds-path /scratch/$user/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
# text-generation-launcher --model-id $model_id --port 9427 --master-port 23457 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
