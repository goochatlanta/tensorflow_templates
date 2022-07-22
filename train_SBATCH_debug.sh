#!/bin/bash
#SBATCH --job-name=brian.harrinton
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=titans-out-%j.txt
#SBATCH --partition=beards


. /etc/profile

module load lang/miniconda3/4.10.3

source activate cs4321

python -m debugpy --wait-for-client --listen 0.0.0.0:54321 --log-to ./logs_debugpy /home/brian.harrington/cs4321/code/tensorflow_templates/trainer/task.py \
--model_dir="/home/brian.harrington/cs4321/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"