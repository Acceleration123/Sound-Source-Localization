# Sound-Source-Localization
## Conventional Algorithms
In this repository, three kinds of conventional algorithms for SSL(Sound Source Localizaition) task are provided:

* TDOA: [GCC-PHAT](https://ieeexplore.ieee.org/document/1162830)
* SRP: a version using DS beam
* Signal Subspace Family: [MUSIC](https://ieeexplore.ieee.org/document/1143830) and [ESPRIT](https://ieeexplore.ieee.org/document/32276) algorithms

Some RIRs with only one speaker under different conditions are provided in the folder named single_speaker_rir and you can also generate different RIRs through [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) according to your own need.

## SSL Based on Deep Learning
Besides, an unofficial implementation of multi-task U-Net, a neural network that predicts tf-bins where direct speech signal occupies a dominant position, can be found here. See more details here: https://arxiv.org/abs/2005.04376

You can run rir_generator.py to generate RIRs first according to your needs and save them as training data. We recommend you to generate a fixed validation dataset first exploiting dataset_generator.py for the training stage.
