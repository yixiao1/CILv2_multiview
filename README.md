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
 * Train/evaluate (offline) your own trained CIL++ models
 * Test your models on CARLA 0.9.13

-------------------------------------------------------------
### Environment Setup

* Python version: 3.8
* Cuda version: 11.6
* Please find more information about the packages in our [requirements.txt](https://github.com/yixiao1/CILv2_multiview/blob/main/requirements.txt)

1. Set up the conda environment for the experiments:

        conda create --name CILv2Env python=3.8
        conda activate CILv2Env

2. Download [CARLA 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13/) to your root directory and build up CARLA docker:

        export ROOTDIR=<Path to your root directory>
        cd $ROOTDIR
        export CARLAPATH=$ROOTDIR/CARLA_0.9.13/PythonAPI/carla/:$ROOTDIR/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

   For using CARLA docker, you can either pull or build up the container:

        To pull, run: docker pull carlasim/carla:0.9.13
        To build up, run: docker image build -f $ROOTDIR/CARLA_0.9.13/Dockerfile -t CARLA0913 $ROOTDIR/CARLA_0.9.13/

3. Download the CIL++ repository in your root directory:

        cd $ROOTDIR
        git clone https://github.com/yixiao1/CILv2_multiview.git
        cd $ROOTDIR/CILv2_multiview

4. Define environment variables

        export PYTHONPATH=$CARLAPATH:$ROOTDIR/CILv2_multiview:$ROOTDIR/CILv2_multiview/run_CARLA_driving:$ROOTDIR/CILv2_multiview/scenario_runner
        export TRAINING_RESULTS_ROOT=<Path to the directory where the results to be saved>
        export DATASET_PATH=<Path to the directory where the datasets are stored>
        export SENSOR_SAVE_PATH=<Path to the directory where the driving test frames are stored>
        export DRIVING_TEST_ROOT=$ROOTDIR/CILv2_multiview/run_CARLA_driving/

5. Install the required packages:

        conda install --file requirements.txt

-------------------------------------------------------------
### Benchmark our trained CIL++

1. Download our trained [CIL++ model]() to your `TRAINING_RESULTS_ROOT` directory:

        cd $TRAINING_RESULTS_ROOT
        wget "TBA"

2. Benchmark our trained CIL++:

        cd $DRIVING_TEST_ROOT
        run ./scripts/run_evaluation/CILv2/nocrash_newweathertown_Town02.sh

-------------------------------------------------------------
### Dataset Collection with Roach RL expert

Our code used for dataset collection is built based on the framework from [this repository](https://github.com/zhejz/carla-roach):

-------------------------------------------------------------
### Training & performing offline evaluation on new trained CIL++ models

 * You need to define a configuration file for training. Please refer to [this file](https://github.com/yixiao1/CILv2_multiview/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) in `configs` folder as example

 * Run the main.py file:

        python main.py --process-type train_val --gpus 0 --folder CILv2 --exp CILv2_3cam_smalltest

where `--process-type` defines the process type (could be either train_val or val_only), `--gpus` defines the gpus to be used,
`--folder` is the [configuration folder name](https://github.com/yixiao1/CILv2_multiview/tree/main/configs/CILv2),
and `--exp` is the [configuration yaml file name](https://github.com/yixiao1/CILv2_multiview/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml).
Your results will be saved in $TRAINING_RESULTS_ROOT/_results/<folder_name>/<exp_name>/

-------------------------------------------------------------
### Test your models on CARLA simulator

1. Please make sure that your models are saved in the proper pattern as the downloaded CIL++ model:

        cd $TRAINING_RESULTS_ROOT/_results/<folder_name>/<exp_name>/

    where `folder_name` the the experiment folder name, and `exp_name` is the configuration file name.
    Your models are all saved in ./checkpoints/

2. Define a config file for the benchmarking.

        cd $TRAINING_RESULTS_ROOT/_results/<folder_name>/<exp_name>
        > config45.json

    In the json file, you need to define the model/checkpoint to be tested:

            {
                "agent_name": "CILv2",
                "checkpoint": 45,
                "yaml": "CILv2.yaml"
            }
    where `checkpoint` indicates the checkpoint to be tested, `yaml` is the training configuration file which was automatically generated during training

3. Benchmark your model:

    Notice that to benchmark your own trained models, you need to modify the [script]() by changing the `--agent-config`

        cd $DRIVING_TEST_ROOT
        run ./scripts/run_evaluation/CILv2/nocrash_newweathertown_Town02.sh

-------------------------------------------------------------
### License
This software is released under a XXX license, which allows personal and research use only.
For a commercial license, please contact the authors.

Portions of source code taken from external sources are annotated with links to original files and their corresponding licenses.

-------------------------------------------------------------
### Acknowledgements
* TBA




