<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# Bottom-up (and top-down) attention for image captioning and visual question answering

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)
![Primary language](https://img.shields.io/github/languages/top/best-of-acrv/bottom-up-attention)
[![PyPI package](https://img.shields.io/pypi/pyversions/bottom-up-attention)](https://pypi.org/project/bottom-up-attention/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/bottom_up_attention.svg)](https://anaconda.org/conda-forge/bottom_up_attention)
[![Conda Recipe](https://img.shields.io/badge/recipe-bottom_up_attention-green.svg)](https://anaconda.org/conda-forge/bottom_up_attention)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/bottom_up_attention.svg)](https://anaconda.org/conda-forge/bottom_up_attention)
[![License](https://img.shields.io/github/license/best-of-acrv/bottom-up-attention)](./LICENSE.txt)

Bottom-up attention is a mechanism that improves the performance of tradition image captioning and visual question answering (VQA) approaches, which employ top-down attention. We use bottom-up attention to propose image regions, each with an associated feature vector, while top-down attention is still used in determining feature weightings.

TODO: image of the system's output

The repository contains an open-source implementation of our bottom-up attention algorithm in Python, with access to pre-trained weights for both captioning and VQA tasks. The package provides PyTorch implementations for using training, evaluation, and prediction in your own systems. The package is easily installable with `conda`, and can also be installed via `pip` if you prefer manually managing system dependencies.

Our code is free to use, and licensed under GPL-3. We simply ask that you [cite our work](#citing-our-work) if you use bottom-up attention in your own research.

# Related resources

This repository brings the work from a number of sources together. Please see the links below for further details:

- our original paper: ["Bottom-up and top-down attention for image captioning and visual question answering"](#citing-our-work)
- an efficient open source implementation from CMU: [https://github.com/hengyuan-hu/bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa)
- a more recent PyTorch open source implementation: [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
- the original Caffe implementation: [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

## Installing Bottom-up attention

We offer three methods for installing bottom-up attention:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs bottom-up attention and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. We provide Conda packages through [Conda Forge](https://conda-forge.org/), which recommends adding their channel globally with strict priority:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once you have access to the `conda-forge` channel, bottom-up attention is installed by running the following from inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
u@pc:~$ conda install bottom_up_attention
```

We don't explicitly lock the PyTorch installation to a CUDA-enabled version to maximise compatibility with our users' possible setups. If you wish to ensure a CUDA-enabled PyTorch is installed, please use the following installation line instead:

```
u@pc:~$ conda install pytorch=*=*cuda* bottom_up_attention
```

You can see a list of our Conda dependencies in the [bottom-up attention feedstock's recipe](https://github.com/conda-forge/bottom_up_attention-feedstock/blob/master/recipe/meta.yaml).

### Pip

Before installing via `pip`, you must have the following system dependencies installed if you want CUDA acceleration:

TODO: confirm this is correct

- NVIDIA drivers
- CUDA

Then bottom-up attention, and all its Python dependencies can be installed via:

```
u@pc:~$ pip install bottom_up_attention
```

TODO something about the building of custom layers with CUDA...

### From source

Installing from source is very similar to the `pip` method above

TODO validate this statement is actually true "due to bottom-up attention only containing Python code".

Simply clone the repository, enter the directory, and install via `pip`:

```
u@pc:~$ pip install -e .
```

TODO check this actually handles building of the custom layers with CUDA

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

We also include scripts in the `./scripts` directory to support running bottom-up attention without any `pip` installation, but this workflow means you need to handle all system and Python dependencies manually.

## Using bottom-up attention

### Bottom-up attention from the command line

### Bottom-up attention Python API

## Citing our work

If using bottom-up attention in your work, please cite [our original CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf):

```bibtex
@inproceedings{anderson2018bottom,
  title={Bottom-up and top-down attention for image captioning and visual question answering},
  author={Anderson, Peter and He, Xiaodong and Buehler, Chris and Teney, Damien and Johnson, Mark and Gould, Stephen and Zhang, Lei},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6077--6086},
  year={2018}
}
```

TODO delete everything from here down when finished

### Install NLG-EVAL

For evaluating captioning models, install NLG-EVAL:

```
$ pip install git+https://github.com/Maluuba/nlg-eval.git@master
$ nlg-eval --setup
```

### Data Setup

For the datasets required for this project, please refer to the [Best-Of-ACRV repository](https://github.com/best-of-acrv/acrv-datasets).
Use this repository to download and prepare the COCO and GloVe datasets required for this project.
The data directories should appear in the following structure:

```
root_dir
|--- deploy.py
|--- eval.py
|--- train.py
acrv-datasets
|--- datasets
|------- coco
|------- glove
|------- trainval36
```

`trainval36` are the 36 features per image set provided by the original authors (refer to [this repository](https://github.com/peteanderson80/bottom-up-attention)).
After all data has been downloaded, process the data into the correct format for captioning and visual question answering tasks
using `process.sh`, located in the root directory of this repository.

```
$ sh process.sh
```

## Evaluation

To evaluate with one of the pretrained models, run `eval.py`. There are pretrained models for both captioning and
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

## Training

To train your own model, run `train.py`.

Use `--task` to choose between `captioning` for image captioning and `vqa` for visual question answering.
For example, to train on the image captioning task, run the following command from the root directory of this repository:

```
$ python train.py --task=captioning
```
