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

Train and evaluate an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/train_ERM.py
```

The model will be saved in `./AAAIcodeSubmissoin__model_sensitivity_map/save_dir`.


Measure the sensitivity map of an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/model_sensitivity_map.py
```

### Train model with Spectral Adversarial Data Augmentation (SADA)

Train and evaluate the models with SADA by

```bash
python ./AAAIcodeSubmissoin__SADA/train_SADA.py
```

The key SADA data augmentation module is in 

```
./AAAIcodeSubmissoin__SADA/sada.py
```

Attack settings for all datasets:

- `--epsilon` iteration of the checkpoint to load. #Default: 0.2
- `--step_size` step size of the adversarial attack on the amplitude spectrum. #Default: 0.08
- `--num_steps` batch size of the attack steps. #Default: 5
- `--tau` settings for the early stop acceleration. #Default: 1
- `--tau` settings for the early stop acceleration. #Default: 1
- `--random_init` if init with random perturb. #Default: True
- `--randominit_type` settings for the early stop acceleration. #Default: 1
- `--criterion` loss functions to attack. #Default: 'ce', choices=['ce', 'kl']
- `--direction` neg: standard attack || pos:inverse attack. #Default: neg, choices=['pos','neg']

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

- [AugMix](https://github.com/google-research/augmix)
- [AugMax](https://github.com/VITA-Group/AugMax)
- [GUD](https://github.com/ricvolpi/generalize-unseen-domains)
- [M-ADA](https://github.com/joffery/M-ADA)
- [RandConv](https://github.com/wildphoton/RandConv/)

We would like to thank these authors for sharing their codes.
