# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  
# Created by marko.orescanin@nps.edu on 7/21/20

"""params.py

This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.

"""
from scipy.constants import golden, pi
import tensorflow as tf
import numpy as np
import argparse
import os
import datetime
import yaml


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run training for GestureRecognitionModeling')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--model_type', type=str, default='fully_connected',
                        help="the type of model to use. allowed inputs are fully_connected and cnn")
    parser.add_argument('--regression', type=str2bool, default=False,
                        help="is it a regression model")
    parser.add_argument('--input_shape', type=int, default=(28, 28),
                        help="the type of model to use. allowed inputs are fully_connected and cnn")
    parser.add_argument('--continue_training', type=str2bool, default=False,
                        help="continue training from a checkpoint")
    parser.add_argument('--checkpoint_path', type=str,
                        help="path to the checkpoint to continue training")
    parser.add_argument('--predict', type=str2bool, default=False,
                        help="predict from a checkpoint, use checkpoint flag to pass a model")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="the type of model to use. allowed inputs are fully_connected and cnn")

    parser.add_argument('--num_epochs', type=int, default=2,
                        help="the type of model to use. allowed inputs are fully_connected and cnn")
    parser.add_argument('--batch_size', type=int, default=6,
                        help="the type of model to use. allowed inputs are fully_connected and cnn")

    parser.add_argument("--optimizer", type=str, default="adam",
                        help="specify the optimizer for the model")
    parser.add_argument("--callback_list", type=str, default=None,
                        help="the callbacks to be added")
    parser.add_argument("--activation_fn", type=str, default="relu",
                        help="specify the hidden layer activation function for the model")
    parser.add_argument("--base_learning_rate", type=int, default=0.001,
                        help="specify the base learning rate for the specified optimizer for the model")
    parser.add_argument("--loss_type", type=str, default="categorical_crossentropy",
                        help=" loss type: Options [categorical_crossentropy | binary_crossentropy]")
    parser.add_argument("--eval_metrics", type=str, default=None,
                        help="a list of the metrics of interest, seperated by commas")
    # multi-gpu training arguments
    parser.add_argument('--mgpu_run', type=str2bool, default=False,
                        help="multi gpu run")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="number of gpu's on the machine, juno is 2")
    # multi-processing arguments
    parser.add_argument('--use_multiprocessing', type=str2bool, default=True,
                        help="specifys weather to use use_multiprocessing in .fit_genrator method ")
    parser.add_argument("--workers", type=int, default=6,
                        help="number of CPU's, for my machine 6 workers, for Juno 18")
    return parser.parse_args()


# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hparams():
    """any preprocessing, special handling of the hparams object"""

    parser = make_argparser()
    print(parser)

    return parser


def save_hparams(hparams):
    path_ = os.path.join(hparams.model_dir, 'params.txt')
    hparams_ = vars(hparams)
    with open(path_, 'w') as f:
        for arg in hparams_:
            print(arg, ':', hparams_[arg])
            f.write(arg + ':' + str(hparams_[arg]) + '\n')

    path_ = os.path.join(hparams.model_dir, 'params.yml')
    with open(path_, 'w') as f:
        yaml.dump(hparams_, f,
                  default_flow_style=False)  # save hparams as a yaml, since that's easier to read and use


