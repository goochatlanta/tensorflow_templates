#!/bin/bash
#SBATCH --job-name=donald.peltier
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards
#SBATCH --reservation=cs4321

. /etc/profile

module load lang/miniconda3/4.10.3

source activate py_39

python trainer/task.py \
--model_dir="/home/donald.peltier/smallwork/data/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




