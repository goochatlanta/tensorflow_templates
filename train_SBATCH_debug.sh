#!/bin/bash
#SBATCH --job-name=eric.jimenez
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards
#SBATCH --nodelist=compute-8-5

. /etc/profile

module load lang/miniconda3/4.10.3

source activate myEnv

python -m debugpy --wait-for-client --listen 0.0.0.0:54321 --log-to ./logs_debugpy /home/eric.jimenez/tensorflow_templates/trainer/task.py \
--model_dir="/home/eric.jimenez/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"