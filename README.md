# README #

This repository contains a PyTorch implementation of the work **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering**, 
which is available [here](https://arxiv.org/abs/1707.07998)

> Anderson, Peter, et al. "Bottom-up and top-down attention for image captioning and visual question answering." 
> Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

This repository is designed to provide out-of-the-box functionality for evaluation and training of
bottom-up attention models for both visual question answering and captioning task, 
with as little overhead as possible. Code was adapted from the following repositories 
[here](https://github.com/hengyuan-hu/bottom-up-attention-vqa) and [here](https://https://github.com/poojahira/image-captioning-bottom-up-top-down).

## Setup ##
From the root directory of this repository, to create the Conda environment to run code from this repository:

```
$ conda env create -f requirements.yml
```
This should set up the conda environment with all prerequisites for running this code. Activate this Conda
environment using the following command:
```
$ conda activate pytorch-bua
```

### Install NLG-EVAL ###
For evaluating captioning models, install NLG-EVAL:
```
$ pip install git+https://github.com/Maluuba/nlg-eval.git@master
$ nlg-eval --setup
```

### Data Setup ###

All data should be downloaded to the ```data``` directory of this repository. The easiest way to download the data is to 
run the ```download_images.sh```, ```download_vqa.sh``` and ```download_captioning.sh``` scripts that are located in the 
```data_utils``` directory.

```
$ cd data_utils
$ sh download_images.sh
$ sh download_vqa.sh
$ sh download_captioning.sh 
``` 

The downloaded features are the 36 features per image set provided by the original authors 
(refer to [this](https://github.com/peteanderson80/bottom-up-attention) repository). 
After all data has been downloaded, process the data into the correct format for captioning and visual question answering tasks 
using ```process.sh```, located in the root directory of this repository.

```
$ sh process.sh
``` 

## Evaluation ##
To evaluate with one of the pretrained models, run ```eval.py```. There are pretrained models for both captioning and 
visual question (VQA) answering tasks.

You can specify the desired task (captioning or VQA). For example, to perform captioning with the provided pretrained model, 
run the following command from the root directory:

```
$ python eval.py --task=captioning
```

Pretrained models will be automatically downloaded and stored in the pretrained/models directory. Alternatively, if you wish to load your own pretrained model, you can do this by specifying a load directory (e.g.):

```
$ python eval.py --task=captioning --load_directory=runs/mymodel
```

Will load a pretrained captioning model, from the directory runs/mymodel.


## Training ##
To train your own model, run ```train.py```.

Use ```--task``` to choose between ```captioning``` for image captioning and ```vqa``` for visual question answering. 
For example, to train on the image captioning task, run the following command from the root directory of this repository:

```
$ python train.py --task=captioning
```

