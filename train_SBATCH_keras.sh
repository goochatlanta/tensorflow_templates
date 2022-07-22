#!/bin/bash
#SBATCH --job-name=eric.jimenez
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/eric.jimenez/models/tensorflow_templates/hamming_output
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate myEnv

python trainer/task.py \
--model_dir="/home/eric.jimenez/models/tensorflow_templates/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




