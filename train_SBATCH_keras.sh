#!/bin/bash
#SBATCH --job-name=david.martin
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
<<<<<<< HEAD
#SBATCH --output=titans-out-%j.txt
=======
#SBATCH --output=./logs_hamming/titans-out-%j.txt
>>>>>>> fc6300463e6872cc6a3e3e734573c1b6ea2f7925
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

<<<<<<< HEAD
source activate cs4321

python trainer/task.py \
--model_dir="/home/david.martin/smallwork/tensorflow_templates/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
=======
source activate py39_cs4321

python trainer/task.py \
--model_dir="/home/marko.orescanin/data/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
>>>>>>> fc6300463e6872cc6a3e3e734573c1b6ea2f7925
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




