# Torch Trainer
Implements a wrapper over the process of training PyTorch models to alleviate the need to explicitly write the commands.

This has been adapted from [here](https://github.com/pytorch/vision/tree/master/references/detection) for simpler models then Mask RCNN for instance.

This code is directed for simpler models which do not have complex loss functions and make use of whatever is available in PyTorch out of the box.

## Usage
This code will alleviate having to explicitly state the necessary commands to train a model.

Either clone this repo or copy the contents of the ```trainer.py``` & ```evaluator.py```, instantiate and train!
