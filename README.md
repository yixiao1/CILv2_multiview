# CIL++ with Multi-View Attention Learning
-------------------------------------------------------------

 <img src="Driving_T5.gif" height="350">

### Publications
This is the official code release of the paper:

Yi Xiao, Felipe Codevilla, Diego Porres and Antonio M. Lopez. [Scaling Vision-based End-to-End Driving with Multi-View Attention Learning]().

Please cite our paper if you find this work useful:

         @article{TBA
         }

### Video
Please check our online [video]()

-------------------------------------------------------------
### Summary

In this repository, you could find materials in order to:

 * Benchmark the trained CIL++ model proposed in our paper
 * Collect datasets using Roach RL expert driver from [this paper](https://arxiv.org/abs/2108.08265)
 * Train/evaluate (offline) on your own trained CIL++ models
 * Test your own trained CIL++ models on CARLA 0.9.13

-------------------------------------------------------------
### Environment Setup

* Python version: 3.8
* Cuda version: 11.6
* Please find more information about the packages in our requirements.txt

1. Set up the conda environment for the experiments:

        conda create --name CILv2Env python=3.8
        conda activate CILv2Env

Download CARLA 0.9.13 and build up CARLA docker:
 * export ROOTDIR=<Path to your root directory>
 * cd $ROOTDIR
 * download [CARLA 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13/) from the CARLA website
 * export CARLAPATH=$ROOTDIR/CARLA_0.9.13/PythonAPI/carla/:$ROOTDIR/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
 * For using CARLA docker, you can either directly pull or build up by yourselves:
    pull down: docker pull carlasim/carla:0.9.13
    build up the CARLA docker: docker image build -f $ROOTDIR/CARLA_0.9.13/Dockerfile -t CARLA0913 $ROOTDIR/CARLA_0.9.13/

Download the repository in your root directory:
 * cd $ROOTDIR
 * git clone https://github.com/yixiao1/CILv2_multiview.git
 * cd $ROOTDIR/CILv2_multiview

Define environment variables
 * export PYTHONPATH=$CARLAPATH:$ROOTDIR/CILv2_multiview
 * export TRAINING_RESULTS_ROOT=<Path to the directory where the results are saved>
 * export DATASET_PATH=<Path to the directory where the datasets download>

Install the required packages:
 * conda install --file requirements.txt

-------------------------------------------------------------
### Benchmark our trained CIL++
* TBA

-------------------------------------------------------------
### Dataset Collection with Roach RL expert

To obtain datasets for training and offline evaluation, ...

-------------------------------------------------------------
### Training & performing offline evaluation on new trained CIL++ models

 * You need to define a configuration file for training. Please refer to [this file](https://https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) in `configs` folder as example

 * Run the main.py file:

        python3 main.py --process-type train_val --gpus 0 --folder CILv2 --exp CILv2_3cam_smalltest

where `--process-type` defines the process type (could be either train_val or val_only), `--gpus` defines the gpus to be used,
`--folder` is the experiment folder defined inside [configs folder](https://github.com/yixiao1/CILv2_multiview/tree/main/configs/CILv2),
and `--exp` is the [configuration yaml file](https://github.com/yixiao1/CILv2_multiview/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) defined for training.

-------------------------------------------------------------
### Online driving test on CIL++ models in CARLA simulator

* export PYTHONPATH=/home/yxiao/CARLA_0.9.13/PythonAPI/carla/:/home/yxiao/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/home/yxiao/CILv2_multiview/run_CARLA_driving:/home/yxiao/CILv2_multiview/scenario_runner:/home/yxiao/CILv2_multiview

* export SENSOR_SAVE_PATH=/datatmp/Datasets/yixiao/CARLA/driving_record/

* export DRIVING_TEST_ROOT=/home/yxiao/CILv2_multiview/run_CARLA_driving/

* cd $DRIVING_TEST_ROOT

* run ./scripts/run_evaluation/CILv2/leaderboard_Town05_test.sh

-------------------------------------------------------------
### Acknowledgements
* TBA
