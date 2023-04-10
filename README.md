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
* Run the following command to install the required packages

```bash
conda create -n cilv2 python=3.7
conda activate cilv2
pip3 install -r requirements.txt
```

-------------------------------------------------------------
### Benchmark our trained CIL++
* TBA

-------------------------------------------------------------
### Dataset Collection with Roach RL expert

To obtain datasets for training and offline evaluation, ...

-------------------------------------------------------------
### Training & performing offline evaluation on new CIL++ models

In a command line, run the following whilst in the root directory of this repository:

```bash
export PYTHONPATH=/home/dporres/CARLA_0.9.13/PythonAPI/carla/:/home/dporres/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/datafast/experiments/dporres/CILv2_multiview/run_CARLA_driving:/datafast/experiments/dporres/CILv2_multiview/scenario_runner:/datafast/experiments/dporres/CILv2_multiview
export TRAINING_RESULTS_ROOT=/datafast/experiments/dporres/VisionTFM/
export DATASET_PATH=/datatmp/Datasets/yixiao/CARLA
export SENSOR_SAVE_PATH=/datafast/Datasets/dporres/CARLA/driving_record/
export DRIVING_TEST_ROOT=/datafast/experiments/dporres/CILv2_multiview/run_CARLA_driving/
```

* Define a configuration file for training. Please refer to [the `CILv2_3cam_smalltest.yaml` file](./configs/CILv2/CILv2_3cam_smalltest.yaml) in `configs` folder as example

* Run the `main.py` file (here, we use 2 gpus for training and validation)
  
```bash
python3 main.py --process-type train_val --gpus 0 1 --folder CILv2 --exp CILv2_3cam_smalltest
```

where `--process-type` defines the process type (could be either train_val or val_only), `--gpus` defines the gpus to be used (w.r.t. the `CUDA_VISIBLE_DEVICES` environment variable), `--folder` is the experiment folder defined inside [configs folder](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/tree/main/configs/CILv2), and `--exp` is the [configuration yaml file](https://github.com/yixiao1/Scaling-Self-Supervised-End-to-End-Driving-with-Multi-View-Attention-Learning/blob/main/configs/CILv2/CILv2_3cam_smalltest.yaml) defined for training.

#### TODO:
* [x] Make sure the network initializations is not undoing the pre-trained weights!
* [x] Remove GradCAM from the code during training and validation
* [x] ***New Model Architecture: `CIL_multiview_vit_oneseq`***: Horizontally stack the `cam` views into one image (new width of $3W$), patchifiy this, and feed it as a single sequence to the Encoder
  * *Advantage:* This lets us avoid having to average the output of the Encoder w.r.t. the `[CLS]` token for each view as was done in `CIL_multiview`
  * *Advantage:* We can easily access **where** is the model focusing on to make its decision
  * *Disadvantage:* the sequence length is now `S*cam*(H//P)**2 + 1`, whereas before it was only `(H//P)**2 + 1` per camera view
  * *Disadvantage:* We are forcing it to see like a human, which is in itself limiting
* [ ] Log the attention maps during training and validation
* [ ] ***New Model Architecture: `CIL_multiview_vit_multitokens`***: where we will concatenate the `[CMD]` and `[SPD]` tokens to the sequence instead of element-wise addition (i.e., `x = x + cmd + spd`)
  * We also want to have the action output to be a combination of the `[CMD]` and `[SPD]` tokens, as well as a `[CAM]` token
  * In essence, the `[CAM]` token will be the representation of each camera/view, so we can in essence use its output as a "gate" or weight to perform the action
    * In other words, $\mathbf{a}_t = \sum_{i=1}^{C} w_{i,t} \cdot f(\mathbf{z}_{L,t}^{\texttt{CMD}}, \mathbf{z}_{L,t}^{\texttt{SPD}}, \mathbf{z}_{L,t}^{\texttt{CLS}})$, with $w_{i, t} = g(\mathbf{z}_{L,t}^{\texttt{CAM}_i})$, $f$ and $g$ TBD but most likely an MLP
    * We could treat the weights as logits, i.e., apply the softmax function to them or do a simple normalization: $w_{j,t}=\frac{w_{i,t}}{\sum_{i=1}^{C} w_{i,t}}$
  * *Advantage:* We can easily access **which camera** has the most influence in the model's decision, though this may not happen at all 
* [ ] Recreate Figure 11 of the ViT paper) i.e., the size of attended area by head and network depth/layer; code [here](https://github.com/google-research/vision_transformer/issues/162)

-------------------------------------------------------------
### Online driving test on CIL++ models in CARLA simulator

***NOTE:*** 
* We require the `PYTHONPATH` from the above bash command to be set (i.e., the `PYTHONPATH` from the training/evaluation section). Make sure that the `PYTHONPATH` is set before running the following commands.

So, to run a subset of the `NoCrash` benchmark, run:

```bash
cd $DRIVING_TEST_ROOT
./scripts/run_evaluation/CILv2/smallnocrash_newweathertown_Town02_lbc.sh
```

-------------------------------------------------------------
### Acknowledgements
* TBA
