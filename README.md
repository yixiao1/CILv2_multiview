# CIL++ with Multi-View Attention Learning
-------------------------------------------------------------

 <img src="Driving_T5.gif" height="350">

### Publications
This is the official code release of the paper:

Yi Xiao, Felipe Codevilla, Diego Porres and Antonio M. Lopez. [Scaling Self-Supervised End-to-End Driving with Multi-View Attention Learning]().

Please cite our paper if you find this work useful:

         @article{TBA
         }

### Video
Please check our online [video]()

-------------------------------------------------------------
### Summary

In this repository, you could find materials in order to:

 * Benchmark the trained CIL++ model proposed in our paper
 * Collect datasets using Roach RL expert
 * Train/evaluata (offline) on new CIL++ models
 * Test CIL++ models on CARLA simulator

-------------------------------------------------------------
### Environment Setup

* Download CARLA 0.9.13
* TBA

-------------------------------------------------------------
### Benchmark our trained CIL++
* TBA

-------------------------------------------------------------
### Dataset Collection with Roach RL expert

To obtain datasets for training and offline evaluation, ...

-------------------------------------------------------------
### Training & performing offline evaluation on new CIL++ models

In a command line, run the following whilst

```bash
export PYTHONPATH=/home/dporres/CARLA_0.9.13/PythonAPI/carla/:/home/dporres/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/home/dporres/CILv2_multiview/run_CARLA_driving:/home/dporres/CILv2_multiview/scenario_runner:/home/dporres/CILv2_multiview
export TRAINING_RESULTS_ROOT=/data/yixiao/VisionTFM/
export DATASET_PATH=/datatmp/Datasets/yixiao/CARLA
```

* Define a configuration file for training. Please refer to [the `CILv2_3cam_smalltest.yaml` file](./configs/CILv2/CILv2_3cam_smalltest.yaml) in `configs` folder as example

* Run the `main.py` file (here, we use 2 gpus for training and validation)
  
```bash
python3 main.py --process-type train_val --gpus 0 1 --folder CILv2 --exp CILv2_3cam_smalltest
```

where `--process-type` defines the process type (could be either train_val or val_only), `--gpus` defines the gpus to be used,
`--folder` is the experiment folder defined inside [configs folder](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/tree/main/configs/CILv2),
and `--exp` is the [configuration yaml file](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) defined for training.

-------------------------------------------------------------
### Online driving test on CIL++ models in CARLA simulator

```bash
export PYTHONPATH=/home/yxiao/CARLA_0.9.13/PythonAPI/carla/:/home/yxiao/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/home/yxiao/CILv2_multiview/run_CARLA_driving:/home/yxiao/CILv2_multiview/scenario_runner:/home/yxiao/CILv2_multiview
export SENSOR_SAVE_PATH=/datatmp/Datasets/yixiao/CARLA/driving_record/
export DRIVING_TEST_ROOT=/home/yxiao/CILv2_multiview/run_CARLA_driving/
cd $DRIVING_TEST_ROOT
run ./scripts/run_evaluation/CILv2/leaderboard_Town05_test.sh
```

-------------------------------------------------------------
### Acknowledgements
* TBA
