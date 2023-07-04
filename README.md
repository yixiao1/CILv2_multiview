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

## CoRL 2023 Submission

The goal is to submit for [CoRL 2023](https://www.corl2023.org/), so the deadline is in June 8th. Paper submission opens in May 15th, so start preparing for that!

### TODOs:
* [x] Make sure the network initializations is not undoing the pre-trained weights!
* [x] Remove GradCAM from the code during training and validation
* [x] ***New Model Architecture: `CIL_multiview_vit_oneseq`***: Horizontally stack the `cam` views into one image (new width of $3W$), patchifiy this, and feed it as a single sequence to the Encoder
  * *Advantage:* This lets us avoid having to average the output of the Encoder w.r.t. the `[CLS]` token for each view as was done in `CIL_multiview`
  * *Advantage:* We can easily access **where** is the model focusing on to make its decision
  * *Disadvantage:* the sequence length is now `S*cam*(H//P)**2 + 1`, whereas before it was only `(H//P)**2 + 1` per camera view
  * *Disadvantage:* We are forcing it to see like a human, which is in itself limiting
* [x] Log the attention maps during training and validation
  * [ ] Extract the attention map of any token in the sequence (as there will be more than one)
* [ ] Recreate Figure 11 of the ViT paper) i.e., the size of attended area by head and network depth/layer; code [here](https://github.com/google-research/vision_transformer/issues/162)

### CIL-ViT V1: Multitokens

The idea is simple: add more tokens to the sequence, and the token-patch and patch-patch interactions within the Encoder will deal with the mixing of information computation, rather than our manual tuning of extra 

<img src="https://user-images.githubusercontent.com/24496178/231987260-5e37500e-acbc-413e-bf10-3d4a85c0f290.png" alt= “” width="1000" height="550" title="CIL-ViT V1">

### Ablation Studies for V1:
* Learnable vs. fixed positional embeddings
* Action output: GAP vs. MAP vs. MLP
* Command and speed input: number of layers in MLP
* Special token concatenations: Remove each of the special tokens and see how the model performs
* Image resolution: keep using the pre-trained size of 224x224, but also try 288x288 (highest we can go with the datasets we have)
* Add RandAug and MixUp to the training pipeline
* See longer training curves (i.e., 200 epochs)
* Learning rate warmup/cooldown vs. decay by $\gamma$
* (?) Other pre-trained ViTs (DeiT, ViT-S-16, etc.) or checkpoints (i.e., ImageNet-21k, COCO, etc.), esp for the `[CLS]` token


### Possible additions for V2
* Add the `[CAM]` token to the sequence (pass each camera/view characteristics $\text{cam}_i$ to an MLP to get `[CAM]_i` and then concatenate (?) `[CAM]_i` to the sequence)
* The final action can be the same as before, or let the model decide which camera/view to use (i.e., use the `[CAM]` token as a "gate" or weight to perform the action)
  * This way, we could access **which camera** has the most influence in the model's decision, though this may not happen at all
  * In essence: $\mathbf{a}_{t} = \sum_{i=1}^{C} w_{i,t} \cdot f(\mathbf{z}_{\texttt{[CMD]}^{L,t,i}, \mathbf{z}_{\texttt{[SPD]}^{L,t,i}, \mathbf{z}_{\texttt{[ACT]}^{L,t,i})$, with $w_{i,t} = \text{softmax}(g(\mathbf{z}_{\texttt{[CAM]}^{L,t,i}})$
* (?) Different feature encoding/patchification of the input image/sequence as done in [VIOLA](https://openreview.net/pdf?id=L8hCfhPbFho) (Figure 2)
  * From VIOLA, we could only add the Top K token patches in the whole sequence (w.r.t. the `[ACC]` and `[SPD]` tokens for example) or, as they do, with a Region Proposal Network to get the set of object-related regions in an image, before passing the (shorter) sequence to the Encoder
* (?) Add other type of information as special tokens (e.g., `[SSG]` for the semantic segmentation ground truth, `[LID]` for the LiDAR point cloud, `[DPT]` for the depth, etc.)
* (?) Replaces patches with their augmented versions (oversample the unaugmented ones), as done in [Stabilizing Deep Q-Learning w/ConvNets and ViTs](https://proceedings.neurips.cc/paper/2021/file/1e0f65eb20acbfb27ee05ddc000b50ec-Paper.pdf) (Figure 7 top)
  * This will also let us remove redundant patches and hence speedup the training process

### Possible additions for V3 (?)
* Add time in the form of multiple frames (i.e., $S$ frames) to the sequence to better understand the dynamics (add a `[FLO]` token to the sequence for motion flow?)])
  * Note that [Fast and Precise](https://arxiv.org/abs/2206.00702) proposes that predicting future *states* can be a better training signal than predicting future *actions*
    * The future states could be the next $S$ speeds $s$ and commands $c$ (i.e., $s_1, c_1, s_2, c_2, \dots, s_S, c_S$)
    * Another option is to use a Decoder for this task, but nothing limits us from only focusing on the *power* of tokens in the Encoder

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
