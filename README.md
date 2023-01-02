# Spectral Adversarial Data Augmentation (SADA)

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

This repository includes open-source codes, detailed experimental results and full references of our AAAI-2023 paper 
[*When Neural Networks Fail to Generalize? A Model Sensitivity Perspective*](https://arxiv.org/abs/2212.00850).

## Prerequisites

- Python 3.8
- PyTorch 1.8.1+
- A computing device with GPU

## Getting started

### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/products/distribution)
- Install [CUDA 11.0+](https://developer.nvidia.com/cuda-11.0-download-archive)
- Install [PyTorch 1.8.1+](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.8.1](http://pytorch.org/)

### Dataset & Preparation

All datasets used in our work are publicly available.
In our experiments, we followed the data precrossing in the [RandConv](https://github.com/wildphoton/RandConv/).

- The trained model is at `./work/checkpoints`
- The data splition used in our experiments is at `./work/data/data_info.csv`
- Before running the code, please put the preprocessed images into `./work/data/img`

### Measure the sensitivity map of an ERM model

Train an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/train_ERM.py
```

The model will be saved in `./AAAIcodeSubmissoin__model_sensitivity_map/save_dir`.


Measure the sensitivity map of an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/train_ERM.py
```

### Train model with Spectral Adversarial Data Augmentation (SADA)

Train models with SADA by

```bash
python test_main.py
```

The key SADA data augmentation module is in 

```
```

- `--iter` iteration of the checkpoint to load. #Default: 14500
- `--batch_size` batch size of the parallel test. #Default: 64

## Citation

Please cite these papers in your publications if it helps your research:

```bibtex
@article{zhang2022neural,
  title={When Neural Networks Fail to Generalize? A Model Sensitivity Perspective},
  author={Zhang, Jiajin and Chao, Hanqing and Dhurandhar, Amit and Chen, Pin-Yu and Tajer, Ali and Xu, Yangyang and Yan, Pingkun},
  journal={arXiv preprint arXiv:2212.00850},
  year={2022}
}
```

Link to paper:

- [*When Neural Networks Fail to Generalize? A Model Sensitivity Perspective*](https://arxiv.org/abs/2212.00850).


## Acknowledgement

We directly adopt some codes from the previous works as below:

- [AugMix](https://arxiv.org/abs/2207.05231)
- [RandConv](https://arxiv.org/abs/2207.05231)

We would like to thank these authors.
