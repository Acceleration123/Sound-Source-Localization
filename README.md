# Sound-Source-Localization
In this repository, three kinds of conventional algorithms for SSL(Sound Source Localizaition) task are provided. 

Besides, an unofficial implementation of multi-task U-Net, a neural network that predict tf-bins where direct speech signal occupies a dominant position, is opened.

Paper:https://arxiv.org/abs/2005.04376

You can run rir_generator.py to generate RIRs first according to your needs and save them as training data. We recommend you to generate a fixed validation dataset first exploiting dataset_generator.py
