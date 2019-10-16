# MODL
MODL: Multiclass Object Depth Estimation and Localization

Paper will be published after the 2019 SBR/LARS
This project was made with the base code of the J-MOD2 (https://github.com/isarlab-department-engineering/J-MOD2) [1].

## Installation

OS: Ubuntu 18.04 (other versions were not tested)

Requirements:
1. Keras 2.3 (with Tensorflow backend)
2. Python 3.7
3. OpenCV (tested on version 4.1, older versions SHOULD work as well)

## Models

Training and test code is provided for MODL and the baselines.

MODL trained weights on the Simulated Soccer NAO Dataset (SSNDa) will be available soon.

## Usage: testing on the SSNDa

First, download the SSNDa from [here](https://larocs.ic.unicamp.br/dataset) and *TODO*

## Training

The `train.py` performs training on the SSNDa according to the parameters in the `config.py` file.
You can train different models by simply editing the line `model = MODL(config)` accordingly.

If you wish to train the model on a different dataset, you will probably need to define
new classes in the `Dataset.py`,`SampleType.py`,`DataGenerationStrategy.py` and `Sequence.py` files located in
the `lib` directory to correctly process the data. The model expects RGB inputs 
and depth maps with equal resolution as target. Obstacles labels are provided as 
described [here](https://isar.unipg.it/index.php?option=com_content&view=article&id=53:unrealdataset&catid=17&Itemid=212).

*TODO*

## References
[1] Mancini, Michele, et al. "J-MOD $^{2} $: Joint Monocular Obstacle Detection and Depth Estimation." International Conference on Robotics and Automation (ICRA) 2018.
[Link to paper](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/JMOD2.pdf "Paper PDF")
