#!/bin/bash
#SBATCH --job-name=matthew.akers
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=akers-mnist-test-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.8.3

source activate cs4321

python trainer/task.py \
--model_dir="/h/matthew.akers/GitLab_Repos/tensorflow_templates/MODELS/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




