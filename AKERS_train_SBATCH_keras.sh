#!/bin/bash
#SBATCH --job-name=matthew.akers
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/matthew.akers/models/TF_Templates/hamming_output/akers-mnist-test-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate cs4321

python trainer/task.py \
--model_dir="/home/matthew.akers/models/TF_Templates/" \
--model_type="naive_fc" \
--num_epochs=100 \
--batch_size=32 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




