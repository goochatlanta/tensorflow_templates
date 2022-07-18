#!/bin/bash
# trail_local.sh
# A simple shell script to run template locally
#
# Other than running full jobs locally, the ability
# to do so is important for debugging your code before submitting a job to cluster

python task.py \
--model_dir="/home/jacob.slaughter/trainer/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=10 \
--batch_size=10 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="checkpoint, csv_log"




