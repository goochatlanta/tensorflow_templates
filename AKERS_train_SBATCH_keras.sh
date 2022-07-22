#!/bin/bash
#SBATCH --job-name=matthew.akers
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
<<<<<<< HEAD:AKERS_train_SBATCH_keras.sh
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=akers-mnist-test-out-%j.txt
=======
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
>>>>>>> fc6300463e6872cc6a3e3e734573c1b6ea2f7925:train_SBATCH_keras.sh
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

<<<<<<< HEAD:AKERS_train_SBATCH_keras.sh
source activate cs4321

python trainer/task.py \
--model_dir="/h/matthew.akers/GitLab_Repos/tensorflow_templates/MODELS/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
=======
source activate py39_cs4321

python trainer/task.py \
--model_dir="/home/marko.orescanin/data/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
>>>>>>> fc6300463e6872cc6a3e3e734573c1b6ea2f7925:train_SBATCH_keras.sh
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




