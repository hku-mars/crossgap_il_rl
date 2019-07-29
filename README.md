# Crossgap_IL_RL
## Flying through a narrow gap using neural network: an end-to-end planning and control approach
**Crossgap_IL_RL** is the open-soured project of our IROS_2019 paper "Flying through a narrow gap using neural network: an end-to-end planning and control approach"
(our preprint version on [*arxiv*](https://arxiv.org/abs/1903.09088), our video on [*Youtube*](https://www.youtube.com/watch?v=jU1qRcLdjx0)). Including some of the training codes, pretrain networks, and simulator (based on [*Airsim*](https://github.com/microsoft/AirSim)).

<div align="center">
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/merge.jpg" width = 55.4% />
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/gap_pose_2.jpg" width = 41.6% />
</div>

**Introduction**:
Our project can be divided into two phases, the imitation and reinforcement learning. In the first phase, we train our end-to-end policy network by imitating from a tradition pipeline. In the second phase, we fine-tune our policy network using reinforcement learning to improve the network performance. The framework of our systems is shown as follows.

**Fine-tuning network using reinforcement-learning:**
<div align="center">
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/RL.gif" width = 50% />
</div>

**Our realworld experiments:**
<div align="center">
    <style="margin-left:45px">
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/30.gif" width = 45%/>
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/30_15.gif" width = 45%/>
</div>
<div align="center">
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/45.gif" width = 45%/>
    <img src="https://github.com/hku-mars/crossgap_il_rl/blob/master/pics/60.gif" width = 45%/>
</div>

**Author:** [Jiarong Lin](https://github.com/ziv-lin)

## 1. Prerequisites
### 1.1 TensorFlow and Pytorch(option)
Follow [TensorFLow Installation](https://www.tensorflow.org/install) and [Pytorch Installation](https://pytorch.org/get-started/locally/)

### 1.2 AirSim
Following the tutorial of [Microsoft airsim](https://github.com/microsoft/AirSim), kindly setup your environment.

### 1.3 OpenAI-baseline
Our reinforcement-learning is based on [OpenAI-baseline](https://github.com/openai/baselines) platfrom. However, since we train our network by modifying some of its codes, theirfore, our project include the codes of OpenAI-baseline, which is forked form its [github](https://github.com/openai/baselines).

### 1.4 Python packages
The following package is needed in this project, you can install the following packages by pip, 
base on your python's environment settings. 
* numpy (for matrix computing)
* openCV2 
* transforms3d (for SE3 transformation)
* pickle

### 1.5 (*Option for realworld experiments*) DJI_ROS and DJI_SDK

## 2 Examples
### 2.1 Testing networks
*  Comparison of the trajectory generated from the traditional method and from network.
```
cd python_scripts/test
python net_vs_tr_and_pl.py
```
*  Test loading policy network
```
cd python_scripts/test
python test_policy_net.py
```
### 2.2 Cross a narrow gap using model-based approach.

### 2.3 Imitation-learning
* Imitation learning of motion-planning.
* Imitation learning of SE3 geometry controller.
### 2.4 Reinforment-learning
* Environment setup
* Reinforcement-learning
### 2.5 Real-world experiment.
### 2.6 Acknowledgments
Thanks for Luqi.Wang and [Fei.Gao](https://ustfei.com/), without their contributions, our works can’t be finished as we expected.

## 6. License
The source code is released under [GPLv2](http://www.gnu.org/licenses/) license.

## 7. Notice
Since I have transferred from the Hong Kong University of Science and Technology (HKUST) to the University of Hong Kong (HKU), and our new lab is under construction, therefore this project is paused for several months. Some of the codes in this project might not be well constructed or well testing.
However, we insist on open our code for sharing our discovery, we hope some of our current work can help you.  Thank you~