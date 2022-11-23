# CIL++ with Multi-View Attention Learning
-------------------------------------------------------------

 <img src="Driving_T5.gif" height="350">

### Publications
This is the official code release of the paper:

Yi Xiao, Felipe Codevilla, Diego Porres, Ziyang Hong and Antonio M. Lopez. [Scaling Self-Supervised End-to-End Driving with Multi-View Attention Learning]().

Please cite our paper if you find this work useful:

         @article{TBA
         }

### Video
Please check our online [video]()

-------------------------------------------------------------
### Summary

In this repository, you could find materials for:

 * Benchmarking the trained CIL++ model proposed in our paper
 * Collecting datasets using Roach RL expert
 * Training/evaluating (offline) new CIL++ models
 * Performing online driving test on CIL++ models on CARLA

-------------------------------------------------------------
### Environment Setup

* Download CARLA 0.9.13
* TBA


-------------------------------------------------------------
### Benchmark trained CIL++
* TBA

-------------------------------------------------------------
### Dataset Collection with Roach RL expert

To obtain datasets for training and offline evaluation, ...

-------------------------------------------------------------
### Training & offline evaluation on new CIL++ models

* export TRAINING_RESULTS_ROOT=/data/yixiao/VisionTFM/

* export DATASET_PATH=/datatmp/Datasets/yixiao/CARLA

* Define a configuration file for training. Please refer to [this file](https://https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) in `configs` folder as example

* Run the main.py file with "train_encoder" process:

        python3 main.py --process-type train_val --gpus 0 1 --folder CILv2 --exp CILv2_3cam_smalltest

where `--process-type` defines the process type (could be either train_val or val_only), `--gpus` defines the gpus to be used,
`--folder` is the experiment folder defined inside [configs folder](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/tree/main/configs/CILv2),
and `--exp` is the [configuration yaml file](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) defined for training.

-------------------------------------------------------------
### Online driving test on CIL++ models on CARLA
* TBA

-------------------------------------------------------------
### Acknowledgements
* TBA